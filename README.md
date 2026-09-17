# PrincipalDeskAI — TCEA Prototype

This is the first runnable prototype of a Principal's Desk AI assistant for
Techno College of Engineering Agartala (TCEA).

## What it does

1. Reads PDF and DOCX documents from `documents/`.
2. Splits the text into searchable chunks.
3. Retrieves the most relevant source excerpts for a question.
4. If `OPENAI_API_KEY` is configured, asks an OpenAI model to answer using
   only those excerpts.
5. Shows the source documents used for the answer.

This is deliberately a **read-only knowledge assistant**. It does not:
- approve leave,
- change marks,
- make disciplinary decisions,
- send official emails,
- modify student records,
- make autonomous administrative decisions.

## Included in this first prototype

The `documents/` folder contains non-sensitive institutional material:
- Academic Rules
- Faculty Rules & Regulations Handbook 2026
- Semester Examination & Internal Assessment Regulations 2026
- Student Handbook 2026
- Library Rules
- Health Camp Notice
- Swachhata Abhiyan Memo

Two uploaded sources were intentionally not included:
- the Tiranga Rally memo containing student personal/contact information;
- the scanned Tripura University exam schedule, because it is not reliably
  machine-readable and should not be treated as current without verification.

## Run on Windows

Open Command Prompt in this folder:

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Set your API key:

```text
set OPENAI_API_KEY=YOUR_API_KEY
```

Then run:

```text
streamlit run app.py
```

The browser should open the local application, normally at:

```text
http://localhost:8501
```

## First test questions

Try:

- What is the minimum overall attendance required for SEE?
- What happens if attendance is below 60%?
- What is the CIE/SEE percentage split?
- What are the SGPA/CGPA requirements stated in the academic rules?
- What was the budget mentioned in the Swachhata Abhiyan memo?
- What are the dates and timings in the health camp notice?

## Important source-authority behavior

The uploaded 2026 handbooks explicitly state that they are consolidated/working
documents and that the latest official regulation, ordinance, notification,
appointment/service rule, management circular, or TCEA order prevails where
applicable. The assistant should therefore report authority/currency limitations
rather than silently resolving them.

## Next development stage

After this prototype is tested, the project can be extended with:

- role-based login (Principal / Registrar / HOD / Faculty / Staff);
- separate restricted student-data store;
- attendance and result dashboards;
- notice drafting with approval workflow;
- meeting and task management;
- email drafting/sending only after explicit approval;
- audit log of every AI-generated recommendation/action;
- OCR pipeline for scanned documents;
- PostgreSQL + pgvector for larger knowledge bases.
