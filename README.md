# Kindred organization verification

A FastAPI starter application for organization registration, OCR assisted document checks, evidence summaries, and human admin review.

## Run locally

1. Install Python 3.10 or newer and Tesseract OCR. On macOS, install Tesseract with `brew install tesseract`.
2. Create a virtual environment, then install dependencies:

   ```sh
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set `ADMIN_TOKEN` to a long random secret. Set `GROQ_API_KEY` for both LangGraph agents and AI reports, and `TAVILY_API_KEY` for official-domain web search. The registration form asks NGOs for their Darpan ID separately from their registration number.
4. Start the server:

   ```sh
   .venv/bin/python -m uvicorn app:app --reload
   ```

Open <http://127.0.0.1:8000> for the organization form and <http://127.0.0.1:8000/admin> for the reviewer dashboard. Admin document links require sign in.

Run both the dependency installation and Uvicorn commands through `.venv/bin/python` as shown. This ensures Uvicorn uses the interpreter where `pytesseract` and the other packages were installed. OCR also requires the Tesseract executable on your system path; if it is unavailable, the site still starts and marks document checks as unavailable.

## Behavior

- Accepts up to eight image or PDF files per application (12 MB each). PDFs with selectable text are read directly; scanned PDFs and images use Tesseract through `pytesseract`.
- Compares normalized organization names and registration numbers. Missing fields and unavailable OCR checks are shown explicitly.
- The LangGraph workflow has two sequential agents: the first extracts form and document evidence with Groq; the second searches official government domains through Tavily and uses Groq to summarize possible matches, conflicts, and unavailable checks. It runs on submission and can be rerun from the admin dashboard.
- Configure `GROQ_API_KEY` and `TAVILY_API_KEY` in `.env`. LangGraph runs locally and does not need its own API key. `LANGSMITH_API_KEY` is optional and only enables tracing. If Tavily is not configured, the document agent can still run, but government web search is reported as unconfigured.
- NGO searches prioritize the separately submitted NGO Darpan ID and the official Darpan domain. The public Darpan search is linked in the dashboard; the government portal says a Darpan ID by itself is not an endorsement or recognition.
- Government search results are evidence leads from official-domain web pages, not direct confirmation from registry APIs. NGO Darpan search is linked in the admin review; the final verification decision remains with a human reviewer.
- The workflow sends extracted document text and organization fields to Groq, and organization identity/search terms to Tavily. Common PAN, Aadhaar-like 12-digit IDs, email addresses, and phone numbers are redacted from text sent to Groq; review your data handling and vendor terms before production use.
- Reviewers can verify, request additional information, or reject. A reason is required for the latter two decisions.
- After an admin records a decision, the dashboard offers a PDF final review report with the reviewer reason, document field comparisons, agent findings, official-domain source links and snippets when available, and explicit limitations. Only authenticated admins can download it. The AI does not make rejection decisions, and the report does not claim government confirmation when no source evidence was returned.
- Applications and uploaded files are stored locally under `data/` by default. Set `DATA_DIR` to a persistent location when deploying.
- Admin reviewers can inspect extracted document fields and text, and use **Re-read documents** to process applications submitted before the PDF text extraction update.

## Deployment notes

Serve behind HTTPS, set `COOKIE_SECURE=1`, use a unique `ADMIN_TOKEN`, and configure persistent storage. This starter uses a shared admin token; add individual accounts and role management before a multi reviewer production rollout. Restrict upload access and apply a retention policy appropriate to the documents you collect.
