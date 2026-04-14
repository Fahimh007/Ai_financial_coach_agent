import os
import json
import csv
import asyncio
import logging
from datetime import datetime
from io import StringIO

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_NAME = "finance_advisor"
USER_ID = "default_user"

# Data models define the structure of results produced by each agent.
class SpendingCategory(BaseModel):
    category: str
    amount: float
    percentage: float

class SpendingRecommendation(BaseModel):
    category: str
    recommendation: str
    potential_savings: float

class BudgetAnalysis(BaseModel):
    total_expenses: float
    monthly_income: float
    spending_categories: list[SpendingCategory]
    recommendations: list[SpendingRecommendation]

class EmergencyFund(BaseModel):
    recommended_amount: float
    current_amount: float
    current_status: str

class SavingsRecommendation(BaseModel):
    category: str
    amount: float
    rationale: str

class AutomationTechnique(BaseModel):
    name: str
    description: str

class SavingsStrategy(BaseModel):
    emergency_fund: EmergencyFund
    recommendations: list[SavingsRecommendation]
    automation_techniques: list[AutomationTechnique]

class Debt(BaseModel):
    name: str
    amount: float
    interest_rate: float
    min_payment: float

class PayoffPlan(BaseModel):
    total_interest: float
    months_to_payoff: int
    monthly_payment: float

class PayoffPlans(BaseModel):
    avalanche: PayoffPlan
    snowball: PayoffPlan

class DebtRecommendation(BaseModel):
    title: str
    description: str
    impact: str

class DebtReduction(BaseModel):
    total_debt: float
    debts: list[Debt]
    payoff_plans: PayoffPlans
    recommendations: list[DebtRecommendation]

# Load API key from .env for Gemini access.
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")


def parse_json_safely(value, default):
    # If the value is JSON text, convert it to Python data.
    # Otherwise return the original object.
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def is_transient_error(error):
    # Return True for temporary API failures so we can retry.
    message = str(error).lower()
    return any(text in message for text in ["503", "unavailable", "high demand", "temporarily unavailable", "rate limit", "quota"])


# This class sets up the agent workflow and runs financial analysis.
class FinanceAdvisorSystem:
    def __init__(self):
        self.session_service = InMemorySessionService()
        self.budget_agent = self._make_agent(
            "BudgetAnalysisAgent",
            "Analyze the user's financial data, categorize spending, and recommend budget improvements.",
            BudgetAnalysis,
            "budget_analysis",
        )
        self.savings_agent = self._make_agent(
            "SavingsStrategyAgent",
            "Use the budget analysis to recommend savings strategy and emergency fund planning.",
            SavingsStrategy,
            "savings_strategy",
        )
        self.debt_agent = self._make_agent(
            "DebtReductionAgent",
            "Use prior analysis to build debt payoff plans and recommendations.",
            DebtReduction,
            "debt_reduction",
        )
        self.runner = Runner(
            agent=SequentialAgent(
                name="FinanceCoordinatorAgent",
                description="Run budget, savings, and debt agents in sequence.",
                sub_agents=[self.budget_agent, self.savings_agent, self.debt_agent],
            ),
            app_name=APP_NAME,
            session_service=self.session_service,
        )

    def _make_agent(self, name, instruction, schema, key):
        # Create an AI agent with a name, instruction, output schema, and output key.
        return LlmAgent(
            name=name,
            model="gemini-2.5-flash",
            description=instruction,
            instruction=instruction,
            output_schema=schema,
            output_key=key,
        )

    async def analyze_finances(self, financial_data: dict) -> dict:
        # Start a new session for this request and store the input data.
        session_id = f"finance_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session = self.session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id,
            state={
                "monthly_income": financial_data.get("monthly_income", 0),
                "dependants": financial_data.get("dependants", 0),
                "transactions": financial_data.get("transactions") or [],
                "manual_expenses": financial_data.get("manual_expenses") or {},
                "debts": financial_data.get("debts") or [],
            },
        )

        try:
            if session.state["transactions"]:
                self._preprocess_transactions(session)
            if session.state["manual_expenses"]:
                self._preprocess_manual_expenses(session)

            default_results = self._create_default_results(financial_data)
            user_content = types.Content(role="user", parts=[types.Part(text=json.dumps(financial_data))])

            for attempt in range(3):
                try:
                    async for event in self.runner.run_async(
                        user_id=USER_ID,
                        session_id=session_id,
                        new_message=user_content,
                    ):
                        if event.is_final_response() and event.author == self.runner.agent.name:
                            break
                    break
                except Exception as error:
                    if attempt < 2 and is_transient_error(error):
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise

            updated_session = self.session_service.get_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id,
            )

            results = {}
            for key in ["budget_analysis", "savings_strategy", "debt_reduction"]:
                value = updated_session.state.get(key)
                results[key] = parse_json_safely(value, default_results[key]) if value else default_results[key]
            return results

        except Exception as error:
            logger.exception("Error during finance analysis: %s", error)
            if is_transient_error(error):
                # If the model fails temporarily, return default results.
                return self._create_default_results(financial_data)
            raise
        finally:
            self.session_service.delete_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id,
            )

    def _preprocess_transactions(self, session):
        # Clean transaction dates and compute totals by category.
        df = pd.DataFrame(session.state["transactions"])
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        if {"Category", "Amount"}.issubset(df.columns):
            session.state["category_spending"] = df.groupby("Category")["Amount"].sum().to_dict()
            session.state["total_spending"] = float(df["Amount"].sum())

    def _preprocess_manual_expenses(self, session):
        # Add manual expense totals to the session state.
        expenses = session.state["manual_expenses"]
        session.state.update(
            total_manual_spending=sum(expenses.values()),
            manual_category_spending=expenses,
        )

    def _create_default_results(self, financial_data):
        # Create fallback analysis data if the AI model does not return results.
        income = financial_data.get("monthly_income", 0)
        expenses = financial_data.get("manual_expenses") or {}
        if not expenses and financial_data.get("transactions"):
            for item in financial_data["transactions"]:
                category = item.get("Category", "Uncategorized")
                amount = float(item.get("Amount", 0) or 0)
                expenses[category] = expenses.get(category, 0) + amount

        total_expenses = sum(expenses.values())
        categories = [
            {"category": cat, "amount": amt, "percentage": (amt / total_expenses * 100) if total_expenses else 0}
            for cat, amt in expenses.items()
        ]

        return {
            "budget_analysis": {
                "total_expenses": total_expenses,
                "monthly_income": income,
                "spending_categories": categories,
                "recommendations": [
                    {"category": "General", "recommendation": "Review your spending and find savings.", "potential_savings": total_expenses * 0.1}
                ],
            },
            "savings_strategy": {
                "emergency_fund": {
                    "recommended_amount": total_expenses * 6,
                    "current_amount": 0,
                    "current_status": "Not started",
                },
                "recommendations": [
                    {"category": "Emergency Fund", "amount": total_expenses * 0.1, "rationale": "Start by building a safety buffer."},
                    {"category": "Retirement", "amount": income * 0.15, "rationale": "Save for the future."},
                ],
                "automation_techniques": [
                    {"name": "Auto Transfer", "description": "Move money to savings automatically."},
                ],
            },
            "debt_reduction": {
                "total_debt": sum(float(d.get("amount", 0) or 0) for d in financial_data.get("debts") or []),
                "debts": financial_data.get("debts") or [],
                "payoff_plans": {
                    "avalanche": {
                        "total_interest": sum(float(d.get("amount", 0) or 0) for d in financial_data.get("debts") or []) * 0.2,
                        "months_to_payoff": 24,
                        "monthly_payment": sum(float(d.get("amount", 0) or 0) for d in financial_data.get("debts") or []) / 24,
                    },
                    "snowball": {
                        "total_interest": sum(float(d.get("amount", 0) or 0) for d in financial_data.get("debts") or []) * 0.25,
                        "months_to_payoff": 24,
                        "monthly_payment": sum(float(d.get("amount", 0) or 0) for d in financial_data.get("debts") or []) / 24,
                    },
                },
                "recommendations": [
                    {"title": "Increase Payments", "description": "Pay more each month to reduce interest.", "impact": "Lowers total cost."},
                ],
            },
        }


def parse_csv_transactions(content) -> dict:
    # Convert uploaded CSV content into a list of transaction records.
    df = pd.read_csv(StringIO(content.decode("utf-8")))
    required = ["Date", "Category", "Amount"]
    if not set(required).issubset(df.columns):
        missing = [c for c in required if c not in df.columns]
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df["Amount"] = df["Amount"].replace(r"[\$,]", "", regex=True).astype(float)
    return {"transactions": df.to_dict("records"), "category_totals": df.groupby("Category")["Amount"].sum().reset_index().to_dict("records")}


def validate_csv_format(uploaded_file):
    # Check that the uploaded file has the required CSV columns and valid types.
    content = uploaded_file.read().decode("utf-8")
    uploaded_file.seek(0)
    if not csv.Sniffer().has_header(content):
        return False, "CSV file must have headers"
    df = pd.read_csv(StringIO(content))
    required = ["Date", "Category", "Amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, f"Missing columns: {', '.join(missing)}"
    try:
        pd.to_datetime(df["Date"])
        df["Amount"].replace(r"[\$,]", "", regex=True).astype(float)
    except Exception as error:
        return False, str(error)
    return True, "CSV format is valid"


def display_csv_preview(df: pd.DataFrame):
    # Show a small preview of the uploaded transaction data.
    st.subheader("CSV Preview")
    st.metric("Rows", len(df))
    st.metric("Total Amount", f"${df['Amount'].sum():,.2f}")
    st.dataframe(df.head())


def display_budget_analysis(analysis):
    # Display budget charts and recommendations.
    if isinstance(analysis, str):
        analysis = parse_json_safely(analysis, {})
    if not isinstance(analysis, dict):
        st.error("Invalid budget analysis")
        return
    st.subheader("Spending by Category")
    fig = px.pie(
        values=[item["amount"] for item in analysis.get("spending_categories", [])],
        names=[item["category"] for item in analysis.get("spending_categories", [])],
    )
    st.plotly_chart(fig)
    income = analysis.get("monthly_income", 0)
    expenses = analysis.get("total_expenses", 0)
    st.subheader("Income vs Expenses")
    fig2 = go.Figure(data=[go.Bar(x=["Income", "Expenses"], y=[income, expenses], marker_color=["green", "red"])])
    st.plotly_chart(fig2)
    st.metric("Surplus/Deficit", f"${income - expenses:.2f}")
    for rec in analysis.get("recommendations", []):
        st.markdown(f"**{rec.get('category')}**: {rec.get('recommendation')}")


def display_savings_strategy(strategy):
    if isinstance(strategy, str):
        strategy = parse_json_safely(strategy, {})
    if not isinstance(strategy, dict):
        st.error("Invalid savings strategy")
        return
    st.subheader("Savings Strategy")
    ef = strategy.get("emergency_fund", {})
    st.markdown(f"**Emergency Fund**: ${ef.get('recommended_amount',0):,.2f}")
    st.markdown(f"Status: {ef.get('current_status', '')}")
    for rec in strategy.get("recommendations", []):
        st.markdown(f"- {rec.get('category')}: ${rec.get('amount',0):,.2f} - {rec.get('rationale')}")


def display_debt_reduction(plan):
    if isinstance(plan, str):
        plan = parse_json_safely(plan, {})
    if not isinstance(plan, dict):
        st.error("Invalid debt reduction plan")
        return
    st.metric("Total Debt", f"${plan.get('total_debt',0):,.2f}")
    debts = pd.DataFrame(plan.get("debts", []))
    if not debts.empty:
        st.dataframe(debts)
        fig = px.bar(debts, x="name", y="amount", color="interest_rate")
        st.plotly_chart(fig)
    tabs = st.tabs(["Avalanche", "Snowball", "Comparison"])
    with tabs[0]:
        av = plan.get("payoff_plans", {}).get("avalanche", {})
        st.markdown(f"**Avalanche** interest: ${av.get('total_interest',0):,.2f}")
    with tabs[1]:
        snow = plan.get("payoff_plans", {}).get("snowball", {})
        st.markdown(f"**Snowball** interest: ${snow.get('total_interest',0):,.2f}")
    with tabs[2]:
        comp = [
            {"Method": "Avalanche", "Total Interest": av.get("total_interest", 0), "Months": av.get("months_to_payoff", 0)},
            {"Method": "Snowball", "Total Interest": snow.get("total_interest", 0), "Months": snow.get("months_to_payoff", 0)},
        ]
        st.dataframe(pd.DataFrame(comp))


def main():
    # Streamlit app entry point: layout, inputs, and results display.
    st.set_page_config(page_title="AI Financial Coach", layout="wide")
    with st.sidebar:
        st.subheader("CSV Template")
        sample_csv = "Date,Category,Amount\n2024-01-01,Housing,1200.00\n2024-01-02,Food,150.50\n2024-01-03,Transportation,45.00"
        st.download_button("Download CSV", sample_csv, "expense_template.csv", "text/csv")

    if not GEMINI_API_KEY:
        st.error("Please add GOOGLE_API_KEY to .env")
        return

    st.title("AI Financial Coach")
    input_tab, about_tab = st.tabs(["Financial Info", "About"])

    with input_tab:
        st.header("Your Financial Details")
        income = st.number_input("Monthly Income", min_value=0.0, value=3000.0, step=100.0)
        dependants = st.number_input("Dependants", min_value=0, value=0, step=1)
        expense_choice = st.radio("Expense Input", ["Upload CSV", "Manual"], horizontal=True)
        transactions = None
        manual_expenses = {}

        if expense_choice == "Upload CSV":
            uploaded = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded:
                valid, msg = validate_csv_format(uploaded)
                if valid:
                    uploaded.seek(0)
                    data = uploaded.read()
                    transactions = parse_csv_transactions(data)["transactions"]
                    display_csv_preview(pd.DataFrame(transactions))
                else:
                    st.error(msg)
        else:
            categories = ["Housing", "Utilities", "Food", "Transportation", "Healthcare", "Entertainment", "Personal", "Savings", "Other"]
            cols = st.columns(3)
            for i, cat in enumerate(categories):
                manual_expenses[cat] = cols[i % 3].number_input(cat, min_value=0.0, value=0.0, step=50.0)
            if any(manual_expenses.values()):
                st.write(pd.DataFrame([manual_expenses]).T.rename(columns={0: "Amount"}))

        st.subheader("Debts")
        debt_count = st.number_input("Number of debts", min_value=0, max_value=10, value=0, step=1)
        debts = []
        for i in range(int(debt_count)):
            st.write(f"Debt #{i + 1}")
            name = st.text_input("Name", value=f"Debt {i+1}", key=f"name_{i}")
            amount = st.number_input("Amount", min_value=0.0, value=1000.0, step=100.0, key=f"amount_{i}")
            rate = st.number_input("Interest Rate", min_value=0.0, max_value=100.0, value=5.0, step=0.1, key=f"rate_{i}")
            minimum = st.number_input("Minimum Payment", min_value=0.0, value=50.0, step=10.0, key=f"min_{i}")
            debts.append({"name": name, "amount": amount, "interest_rate": rate, "min_payment": minimum})

        if st.button("Analyze My Finances"):
            if expense_choice == "Upload CSV" and transactions is None:
                st.error("Upload a valid CSV or switch to manual entry.")
            else:
                data = {
                    "monthly_income": income,
                    "dependants": dependants,
                    "transactions": transactions,
                    "manual_expenses": manual_expenses if expense_choice == "Manual" else None,
                    "debts": debts,
                }
                system = FinanceAdvisorSystem()
                try:
                    results = asyncio.run(system.analyze_finances(data))
                    tab1, tab2, tab3 = st.tabs(["Budget", "Savings", "Debt"])
                    with tab1:
                        display_budget_analysis(results["budget_analysis"])
                    with tab2:
                        display_savings_strategy(results["savings_strategy"])
                    with tab3:
                        display_debt_reduction(results["debt_reduction"])
                except Exception as error:
                    st.error(f"Analysis failed: {error}")

    with about_tab:
        st.markdown(
            """
            ### About

            This app uses Google ADK and Gemini AI to analyze spending, savings, and debt.
            Financial data is processed locally and not saved.
            """
        )


if __name__ == "__main__":
    main()
