# Principal Desk AI — Development Roadmap

## Phase 1 — Completed prototype
- Document ingestion
- Local retrieval
- Source-grounded answers
- Read-only UI

## Phase 2 — Institutional search
- OCR for scanned PDFs
- Better chunking and metadata
- PostgreSQL/pgvector
- Document version/effective-date metadata
- "Official / working / superseded" source labels

## Phase 3 — Principal dashboard
- Attendance summary
- Examination/result summary
- Pending approvals
- Notices
- Meetings
- Important deadlines

## Phase 4 — Controlled actions
Every action should use:
Request -> AI draft/recommendation -> human approval -> action -> audit log

Sensitive operations must remain human-approved.
