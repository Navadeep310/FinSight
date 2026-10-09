# FinSight

FinSight is a local financial-report analyzer built with Streamlit, ChromaDB, and
Ollama. It extracts text and tables from PDF reports, indexes page-scoped
chunks, and answers questions with page citations.

> The included PDF reports contain fictional training data only. They are not
> real company filings or investment advice.

## Requirements

- Python 3.13.14 (the app also runs on other compatible Python versions)
- [Ollama](https://ollama.com/) with the `qwen2.5:latest` model

## Setup

From the repository directory, create and activate a virtual environment, then
install dependencies:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull qwen2.5:latest
```

Start Ollama if it is not already running, then launch FinSight:

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, upload one or more PDF reports, and
select **Index documents** before asking a question. The local vector index is
stored in `chroma_db/`, which is intentionally excluded from Git.

## Included sample reports

- `Northstar_Components_Fictional_Report.pdf`
- `Cedar_Retail_Fictional_Report.pdf`

Example question: “Compare FY2024 and FY2023 revenue and net income for
Northstar Components and Cedar Retail.”
