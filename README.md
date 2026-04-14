# AI Financial Coach Agent

A Streamlit-based personal finance advisor that uses Google ADK and Google Gemini AI to analyze budgeting, savings, and debt management.

### Live Demo

- https://ai-financial-coach-agent.onrender.com/

## Overview

This app collects financial inputs from the user and runs a multi-step AI workflow to return personalized recommendations.

### Workflow

1. **User input**
   - Monthly income
   - Dependents
   - Transactions via CSV upload or manual expense entry
   - Debt details

2. **Session creation**
   - The app creates an in-memory session to store user state with `InMemorySessionService`.

3. **Agent orchestration**
   - A `SequentialAgent` runs three sub-agents in order:
     - `BudgetAnalysisAgent` — analyzes spending and generates budget recommendations
     - `SavingsStrategyAgent` — recommends emergency fund and savings automation
     - `DebtReductionAgent` — builds debt payoff plans using avalanche and snowball methods

4. **AI model execution**
   - Each agent uses `LlmAgent` with a defined Pydantic output schema.
   - The app sends the consolidated financial data to Gemini and waits for structured responses.

5. **Fallback handling**
   - If the AI response is missing or cannot be parsed, the app returns default recommendations.
   - The UI warns the user when fallback results are shown.

6. **Results display**
   - Plotly charts and tables show spending categories, income vs expenses, savings strategy, and debt reduction plans.

## Features

- Upload transaction CSV files or enter expense categories manually
- Generate budget recommendations and spending breakdowns
- Estimate emergency fund needs and savings allocations
- Compare debt payoff strategies (avalanche vs snowball)
- Display interactive charts with Plotly and Streamlit

## Requirements

- Python 3.13+
- `google-adk`
- `google-genai`
- `streamlit`
- `pandas`
- `plotly`
- `matplotlib`
- `numpy`
- `python-dotenv`
- `pydantic`

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Fahimh007/Ai_financial_coach_agent.git
   cd Ai_financial_coach_agent
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your Gemini API key:
   ```text
   GOOGLE_API_KEY=your_api_key_here
   ```

## Run the App

```bash
streamlit run ai_financial_coach_agent.py
```

Then open the local Streamlit URL shown in the terminal.

## Deployment

For Render, use a `Web Service` and configure:

- Build command: `pip install -r requirements.txt`
- Start command:
  ```bash
  streamlit run ai_financial_coach_agent.py --server.port $PORT --server.address 0.0.0.0 --server.enableCORS false --server.headless true
  ```
- Set `GOOGLE_API_KEY` as a Render environment variable.

## Notes

- The app does not persist user financial data to disk or database.
- Keep `.env` out of source control.
- ` .venv` should be ignored by `.gitignore`.


