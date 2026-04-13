# AI Financial Coach Agent

A Streamlit-based personal finance advisor that uses Google ADK and Gemini AI to analyze budgeting, savings, and debt management.

## Features

- Upload transaction CSVs or enter monthly expenses manually
- Analyze spending categories and generate budget recommendations
- Provide savings strategy guidance and emergency fund estimates
- Offer debt reduction plans using avalanche and snowball methods
- Render interactive charts with Plotly

## Requirements

- Python 3.13+ (project uses a virtual environment)
- `google-adk`
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

## Usage

- Use the sidebar to download the sample CSV template
- Upload your transaction file or enter expenses manually
- Add debt information to get personalized payoff strategies
- Click **Analyze My Finances** to run the AI-driven analysis

## Notes

- The app uses local processing and does not store submitted financial data
- The `.env` file should not be committed to GitHub
- The `.venv` folder is ignored by `.gitignore`

## Repository

`https://github.com/Fahimh007/Ai_financial_coach_agent`
