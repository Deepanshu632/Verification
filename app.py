from __future__ import annotations

import json
import html
import io
import os
import re
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    import fitz
except ImportError:
    fitz = None
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    Image = None

    class UnidentifiedImageError(Exception):
        pass

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "verification.db"
MAX_FILE_BYTES = 12 * 1024 * 1024
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".pdf"}
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Kindred · Organization verification")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def connect() -> sqlite3.Connection:
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def init_db() -> None:
    with connect() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS applications (
            id TEXT PRIMARY KEY, organization_name TEXT NOT NULL, organization_type TEXT NOT NULL,
            registration_number TEXT, darpan_id TEXT, address TEXT NOT NULL, contact_name TEXT NOT NULL,
            email TEXT NOT NULL, phone TEXT, documents TEXT NOT NULL, extracted TEXT NOT NULL,
            checks TEXT NOT NULL, report TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
            submitted_at TEXT NOT NULL, review_reason TEXT, reviewed_at TEXT,
            agent_status TEXT NOT NULL DEFAULT 'not_run', agent_review TEXT
        )""")
        columns = {row[1] for row in db.execute("PRAGMA table_info(applications)")}
        if "darpan_id" not in columns:
            db.execute("ALTER TABLE applications ADD COLUMN darpan_id TEXT")
        if "agent_status" not in columns:
            db.execute("ALTER TABLE applications ADD COLUMN agent_status TEXT NOT NULL DEFAULT 'not_run'")
        if "agent_review" not in columns:
            db.execute("ALTER TABLE applications ADD COLUMN agent_review TEXT")


init_db()


def normalized(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def extract_document_fields(text: str) -> dict[str, str | None]:
    """Extract labeled fields from PDF text or OCR output; unmatched fields stay empty."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    joined = "\n".join(lines)

    def find(pattern: str) -> str | None:
        for line in lines:
            match = re.search(pattern, line, re.I)
            if match:
                value = match.group(1).strip(" :#-\t\r\n")
                if normalized(value) and normalized(value) not in {"name", "number", "no", "id", "address", "organization", "organisation", "registration"}:
                    return value
        return None

    return {
        "organization_name": find(r"(?:organization\s+name|organisation\s+name|registered\s+name|entity\s+name|name\s+of\s+(?:the\s+)?(?:organization|organisation|society|trust|ngo)|organization|organisation)\s*[:#-]?\s*([^\n]+)"),
        "registration_number": find(r"(?:registration|regn?|reg\.)\s*(?:no\.?|number|#|id)?\s*[:#-]?\s*([A-Z0-9][A-Z0-9/-]{3,})"),
        "darpan_id": find(r"(?:ngo\s*)?(?:darpan\s*)?(?:unique\s+)?id\s*[:#-]?\s*([A-Z]{2}/\d{4}/[A-Z0-9/-]{3,})"),
        "address": find(r"(?:(?:registered\s+)?office\s+|registered\s+)?address\s*[:#-]?\s*([^\n]+)"),
        "dates": ", ".join(re.findall(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b", joined)) or None,
        "raw_text": joined[:12000],
    }


def read_document_text(path: Path, extension: str) -> tuple[str, str | None, str]:
    """Read selectable PDF text directly; use Tesseract for scanned PDFs and images."""
    if extension == ".pdf":
        if fitz is None:
            return "", None, "PyMuPDF is not installed in the Python environment running this app."
        try:
            pdf = fitz.open(path)
            try:
                digital_text = "\n".join(page.get_text("text") for page in pdf).strip()
                if digital_text:
                    return digital_text, "PDF text layer", ""
            finally:
                pdf.close()
        except Exception as exc:
            return "", None, f"The PDF could not be read ({type(exc).__name__})."

    if pytesseract is None:
        return "", None, "pytesseract is missing from the Python environment running this app."
    if Image is None:
        return "", None, "Pillow is missing from the Python environment running this app."
    try:
        if extension == ".pdf":
            pdf = fitz.open(path)
            try:
                pages = []
                for page in pdf:
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                    pages.append(pytesseract.image_to_string(image))
                return "\n".join(pages), "Tesseract OCR", ""
            finally:
                pdf.close()
        with Image.open(path) as image:
            return pytesseract.image_to_string(image.convert("RGB")), "Tesseract OCR", ""
    except pytesseract.TesseractNotFoundError:
        return "", None, "The Tesseract executable is not installed or is not on PATH."
    except (UnidentifiedImageError, OSError, RuntimeError) as exc:
        return "", None, f"The document could not be processed ({type(exc).__name__})."


def analyze_documents(documents: list[dict[str, str]]) -> tuple[dict[str, Any], bool]:
    texts: list[str] = []
    methods: list[str] = []
    errors: list[str] = []
    complete = True
    for document in documents:
        path = UPLOAD_DIR / Path(document["path"]).name
        if not path.is_file():
            complete = False
            errors.append(f"Uploaded file {document['name']} is missing from storage.")
            continue
        text, method, error = read_document_text(path, path.suffix.lower())
        texts.append(text)
        if method and method not in methods:
            methods.append(method)
        if error:
            complete = False
            errors.append(error)
    extracted = extract_document_fields("\n".join(texts))
    extracted["extraction_method"] = ", ".join(methods) if methods else "Unavailable"
    extracted["extraction_error"] = "; ".join(dict.fromkeys(errors)) or None
    return extracted, complete


def compare_fields(form: dict[str, str], extracted: dict[str, Any], ocr_complete: bool) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    fields = [("organization_name", "Organization name"), ("registration_number", "Registration number"), ("address", "Address")]
    if form.get("organization_type") == "ngo":
        fields.insert(2, ("darpan_id", "NGO Darpan ID"))
    for key, label in fields:
        submitted = (form.get(key) or "").strip()
        found = (extracted.get(key) or "").strip()
        if not submitted:
            checks.append({"field": label, "result": "missing", "detail": "No value was submitted on the form."})
        elif not found:
            if ocr_complete:
                detail = "No matching field was found in the extracted document text."
            else:
                detail = extracted.get("extraction_error") or "Document text extraction could not be completed."
            checks.append({"field": label, "result": "unavailable", "detail": detail})
        elif not normalized(found):
            checks.append({"field": label, "result": "unavailable", "detail": "A field label was detected, but no readable value followed it in the document text."})
        elif normalized(submitted) == normalized(found):
            checks.append({"field": label, "result": "match", "detail": f"Form: {submitted} · Document: {found}"})
        else:
            checks.append({"field": label, "result": "mismatch", "detail": f"Form: {submitted} · Document: {found}"})
    return checks


def merge_agent_extraction(extracted: dict[str, Any], agent_review: dict[str, Any]) -> dict[str, Any]:
    """Use agent values only when the claimed value is present in the OCR text."""
    result = dict(extracted)
    candidate = agent_review.get("extraction") or {}
    raw = str(result.get("raw_text") or "")
    raw_normalized = normalized(raw)
    for key in ("organization_name", "registration_number", "darpan_id", "address"):
        value = candidate.get(key)
        if not result.get(key) and isinstance(value, str) and value.strip():
            value = value.strip()
            if normalized(value) and normalized(value) in raw_normalized:
                result[key] = value
    dates = candidate.get("dates")
    if not result.get("dates") and isinstance(dates, list):
        found_dates = [str(value).strip() for value in dates if str(value).strip() and normalized(str(value)) in raw_normalized]
        if found_dates:
            result["dates"] = ", ".join(dict.fromkeys(found_dates))
    return result


def make_report(form: dict[str, str], extracted: dict[str, Any], checks: list[dict[str, str]], ocr_complete: bool) -> str:
    fallback = [
        f"{form['organization_name']} submitted a {form['organization_type']} verification application.",
        f"The document text extraction {'completed' if ocr_complete else 'could not be completed'}.",
        "Findings: " + ("; ".join(f"{c['field']}: {c['result']} ({c['detail']})" for c in checks) or "No field comparisons were available."),
        "Document dates detected: " + (extracted.get("dates") or "none detected"),
        "This is an evidence summary, not an approval decision. An admin should review the original documents and resolve unavailable checks.",
    ]
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "\n\n".join(fallback)
    try:
        import httpx
        payload = {
            "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "Explain organization verification evidence in simple language. Only state findings present in the supplied data. Separate mismatches, missing information, and checks that could not be completed. Never recommend automatic approval or rejection."},
                {"role": "user", "content": json.dumps({"submitted_form": form, "extracted_fields": {k: v for k, v in extracted.items() if k != "raw_text"}, "checks": checks, "ocr_completed": ocr_complete})},
            ],
        }
        response = httpx.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"].strip()
        return result or "\n\n".join(fallback)
    except Exception as exc:
        return "\n\n".join(fallback + [f"Groq report generation was unavailable ({type(exc).__name__}); showing the local evidence summary."])


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    for key in ("documents", "extracted", "checks"):
        item[key] = json.loads(item[key])
    item["agent_review"] = json.loads(item["agent_review"]) if item.get("agent_review") else None
    return item


def build_review_pdf(item: dict[str, Any]) -> bytes:
    """Build a downloadable, evidence-linked PDF for a completed admin decision."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    styles = getSampleStyleSheet()
    font_name = "Helvetica"
    font_bold = "Helvetica-Bold"
    font_candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    font_path = next((path for path in font_candidates if path.is_file()), None)
    if font_path:
        try:
            pdfmetrics.registerFont(TTFont("ReviewUnicode", str(font_path)))
            font_name = font_bold = "ReviewUnicode"
        except Exception:
            pass
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName=font_bold, fontSize=20, leading=24, textColor=colors.HexColor("#173f38"), alignment=TA_LEFT, spaceAfter=5))
    styles.add(ParagraphStyle(name="ReportSection", parent=styles["Heading2"], fontName=font_bold, fontSize=12, leading=15, textColor=colors.HexColor("#176b55"), spaceBefore=12, spaceAfter=5))
    styles.add(ParagraphStyle(name="ReportBody", parent=styles["BodyText"], fontName=font_name, fontSize=9, leading=13, textColor=colors.HexColor("#283b38"), spaceAfter=4, wordWrap="CJK"))
    styles.add(ParagraphStyle(name="ReportSmall", parent=styles["BodyText"], fontName=font_name, fontSize=8, leading=11, textColor=colors.HexColor("#52615e"), spaceAfter=3, wordWrap="CJK"))

    def para(value: Any, style: str = "ReportBody") -> Paragraph:
        safe = html.escape(str(value if value is not None else ""), quote=True).replace("\n", "<br/>")
        return Paragraph(safe or "-", styles[style])

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#dce6e2"))
        canvas.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#71817d"))
        canvas.drawString(18 * mm, 9 * mm, "Kindred | Confidential admin review")
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()

    story = [para("Organization verification", "ReportTitle"), para("Final admin review report", "ReportSmall"), Spacer(1, 5)]
    decision = str(item.get("status", "pending")).replace("_", " ").title()
    overview = [
        [para("Organization", "ReportSmall"), para(item.get("organization_name"))],
        [para("Type", "ReportSmall"), para(str(item.get("organization_type", "")).title())],
        [para("Application reference", "ReportSmall"), para(item.get("id"))],
        [para("Decision", "ReportSmall"), para(decision)],
        [para("Reviewed at", "ReportSmall"), para(item.get("reviewed_at") or "Not recorded")],
    ]
    story.append(Table(overview, colWidths=[42 * mm, 130 * mm], style=TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f6f3")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#dce6e2")),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e5ece9")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ])))

    story.append(para("Reviewer rationale", "ReportSection"))
    reason = item.get("review_reason") or "No reviewer note was recorded."
    story.append(para(reason))
    if decision == "Rejected":
        story.append(para("This is the admin reviewer’s stated reason. The AI agent does not make rejection decisions; any supporting evidence and its limitations are listed below.", "ReportSmall"))

    story.append(para("Submitted organization details", "ReportSection"))
    details = [
        ("Registration number", item.get("registration_number") or "Not provided"),
        ("NGO Darpan ID", item.get("darpan_id") or "Not provided"),
        ("Address", item.get("address") or "Not provided"),
        ("Contact", " · ".join(str(item.get(key)) for key in ("contact_name", "email", "phone") if item.get(key)) or "Not provided"),
    ]
    for label, value in details:
        story.append(para(f"{label}: {value}"))

    story.append(para("Document field checks", "ReportSection"))
    checks = item.get("checks") or []
    if checks:
        rows = [[para("Field", "ReportSmall"), para("Result", "ReportSmall"), para("Details", "ReportSmall")]]
        rows.extend([[para(check.get("field"), "ReportSmall"), para(check.get("result"), "ReportSmall"), para(check.get("detail"), "ReportSmall")] for check in checks])
        story.append(Table(rows, colWidths=[38 * mm, 27 * mm, 107 * mm], repeatRows=1, style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f6f3")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#dce6e2")),
            ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e5ece9")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])))
    else:
        story.append(para("No document field checks were recorded."))

    review = item.get("agent_review") or {}
    story.append(para("AI and government-source findings", "ReportSection"))
    story.append(para(f"Agent status: {review.get('status', item.get('agent_status', 'not_run'))}"))
    if review.get("summary"):
        story.append(para(review["summary"]))
    if review.get("review"):
        story.append(para(review["review"].get("summary", "")))
        for conflict in review["review"].get("conflicts", []):
            story.append(para(f"Conflict reported by agent: {conflict}", "ReportSmall"))
        for unavailable in review["review"].get("checks_not_completed", []):
            story.append(para(f"Check not completed: {unavailable}", "ReportSmall"))

    sources = []
    for source in review.get("sources", []):
        url = str(source.get("url", ""))
        host = (urlparse(url).hostname or "").lower()
        if url.startswith("https://") and (host.endswith(".gov.in") or host.endswith(".nic.in")):
            sources.append(source)
    if sources:
        for index, source in enumerate(sources, 1):
            title = html.escape(str(source.get("title") or f"Official source {index}"))
            url = html.escape(str(source["url"]), quote=True)
            story.append(Paragraph(f'<b>Source {index}: {title}</b><br/><link href="{url}" color="#176b55">{url}</link>', styles["ReportSmall"]))
            if source.get("content"):
                story.append(para(source["content"], "ReportSmall"))
    else:
        story.append(para("No official-domain source results were recorded for this application. The report therefore does not claim that a government source independently confirms the reviewer’s reason.", "ReportSmall"))

    story.append(para("Limitations", "ReportSection"))
    story.append(para("OCR may omit or misread document text. Government search results are source leads, not direct registry API confirmation. A missing search result does not prove an organization is unregistered. The final decision recorded here belongs to the human admin reviewer.", "ReportSmall"))
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=17 * mm, bottomMargin=20 * mm, title=f"Organization review - {item.get('organization_name', 'Application')}", author="Kindred")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="apply.html", context={})


@app.post("/applications")
async def submit_application(
    organization_name: str = Form(...), organization_type: str = Form(...),
    registration_number: str = Form(""), darpan_id: str = Form(""), address: str = Form(...),
    contact_name: str = Form(...), email: str = Form(...), phone: str = Form(""),
    documents: list[UploadFile] = File(...),
):
    organization_type = organization_type.strip().lower()
    if organization_type not in {"ngo", "restaurant", "hotel", "hostel"}:
        raise HTTPException(400, "Choose a valid organization type.")
    if not documents or len(documents) > 8:
        raise HTTPException(400, "Upload between 1 and 8 supporting images.")
    app_id = secrets.token_urlsafe(9)
    stored: list[dict[str, str]] = []
    try:
        for upload in documents:
            ext = Path(upload.filename or "").suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(400, f"{upload.filename}: upload a supported image or PDF file.")
            data = await upload.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise HTTPException(413, f"{upload.filename}: maximum file size is 12 MB.")
            safe_name = f"{app_id}_{secrets.token_hex(4)}{ext}"
            path = UPLOAD_DIR / safe_name
            path.write_bytes(data)
            stored.append({"name": upload.filename or safe_name, "path": safe_name})
    except Exception:
        for file in stored:
            (UPLOAD_DIR / file["path"]).unlink(missing_ok=True)
        raise

    extracted, ocr_complete = analyze_documents(stored)
    form = {"organization_name": organization_name.strip(), "organization_type": organization_type, "registration_number": registration_number.strip(), "darpan_id": darpan_id.strip(), "address": address.strip(), "contact_name": contact_name.strip(), "email": email.strip(), "phone": phone.strip()}
    from agents import run_verification_agents
    agent_review = run_verification_agents(form, extracted)
    extracted = merge_agent_extraction(extracted, agent_review)
    checks = compare_fields(form, extracted, ocr_complete)
    report = make_report(form, extracted, checks, ocr_complete)
    agent_status = agent_review.get("status", "failed")
    with connect() as db:
        db.execute("""INSERT INTO applications (id, organization_name, organization_type, registration_number, darpan_id, address, contact_name, email, phone, documents, extracted, checks, report, submitted_at, agent_status, agent_review)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (app_id, form["organization_name"], organization_type, form["registration_number"], form["darpan_id"], form["address"], form["contact_name"], form["email"], form["phone"], json.dumps(stored), json.dumps(extracted), json.dumps(checks), report, datetime.now(timezone.utc).isoformat(), agent_status, json.dumps(agent_review)))
    return RedirectResponse(f"/submitted/{app_id}", status_code=303)


@app.get("/submitted/{app_id}", response_class=HTMLResponse)
def submitted(request: Request, app_id: str):
    with connect() as db:
        row = db.execute("SELECT * FROM applications WHERE id = ?", (app_id,)).fetchone()
    if not row:
        raise HTTPException(404)
    return templates.TemplateResponse(request=request, name="submitted.html", context={"item": row_to_dict(row)})


def require_admin(request: Request) -> bool:
    token = os.getenv("ADMIN_TOKEN")
    return bool(token and request.cookies.get("admin_token") == token)


@app.get("/admin/files/{filename}")
def admin_file(request: Request, filename: str):
    if not require_admin(request):
        return RedirectResponse("/admin/login", status_code=303)
    safe_name = Path(filename).name
    with connect() as db:
        rows = db.execute("SELECT documents FROM applications").fetchall()
    allowed = {doc["path"] for row in rows for doc in json.loads(row["documents"])}
    path = UPLOAD_DIR / safe_name
    if safe_name not in allowed or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, filename=next((doc["name"] for row in rows for doc in json.loads(row["documents"]) if doc["path"] == safe_name), safe_name))


@app.get("/admin/{app_id}/report.pdf")
def download_review_report(request: Request, app_id: str):
    if not require_admin(request):
        return RedirectResponse("/admin/login", status_code=303)
    with connect() as db:
        row = db.execute("SELECT * FROM applications WHERE id = ?", (app_id,)).fetchone()
    if not row:
        raise HTTPException(404)
    item = row_to_dict(row)
    if item["status"] == "pending":
        raise HTTPException(409, "Complete the admin review before downloading the final report.")
    pdf = build_review_pdf(item)
    filename = f"organization-review-{re.sub(r'[^A-Za-z0-9_-]+', '-', item['id']).strip('-')}.pdf"
    return StreamingResponse(io.BytesIO(pdf), media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{filename}"', "Cache-Control": "no-store"})


@app.get("/admin", response_class=HTMLResponse)
def dashboard(request: Request, status: str = "all"):
    if not require_admin(request):
        return RedirectResponse("/admin/login", status_code=303)
    with connect() as db:
        rows = db.execute("SELECT * FROM applications ORDER BY submitted_at DESC").fetchall()
    items = [row_to_dict(row) for row in rows]
    if status in {"pending", "verified", "info_requested", "rejected"}:
        items = [item for item in items if item["status"] == status]
    stats = {key: sum(1 for row in rows if row["status"] == key) for key in ("pending", "verified", "info_requested", "rejected")}
    return templates.TemplateResponse(request=request, name="admin.html", context={"items": items, "stats": stats, "filter": status})


@app.post("/admin/{app_id}/review")
def review(request: Request, app_id: str, decision: str = Form(...), reason: str = Form("")):
    if not require_admin(request):
        return RedirectResponse("/admin/login", status_code=303)
    if decision not in {"verified", "info_requested", "rejected"}:
        raise HTTPException(400, "Invalid review decision")
    if decision in {"info_requested", "rejected"} and not reason.strip():
        raise HTTPException(400, "Add a reason for this decision")
    with connect() as db:
        cursor = db.execute("UPDATE applications SET status = ?, review_reason = ?, reviewed_at = ? WHERE id = ?", (decision, reason.strip(), datetime.now(timezone.utc).isoformat(), app_id))
        if cursor.rowcount == 0:
            raise HTTPException(404)
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/{app_id}/reprocess")
def reprocess_documents(request: Request, app_id: str):
    if not require_admin(request):
        return RedirectResponse("/admin/login", status_code=303)
    with connect() as db:
        row = db.execute("SELECT * FROM applications WHERE id = ?", (app_id,)).fetchone()
        if not row:
            raise HTTPException(404)
        item = row_to_dict(row)
        extracted, extraction_complete = analyze_documents(item["documents"])
        form_data = {key: item[key] or "" for key in ("organization_name", "organization_type", "registration_number", "darpan_id", "address", "contact_name", "email", "phone")}
        checks = compare_fields(form_data, extracted, extraction_complete)
        report = make_report(form_data, extracted, checks, extraction_complete)
        db.execute("UPDATE applications SET extracted = ?, checks = ?, report = ? WHERE id = ?", (json.dumps(extracted), json.dumps(checks), report, app_id))
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/{app_id}/agent-review")
def run_agent_review(request: Request, app_id: str):
    if not require_admin(request):
        return RedirectResponse("/admin/login", status_code=303)
    from agents import run_verification_agents
    with connect() as db:
        row = db.execute("SELECT * FROM applications WHERE id = ?", (app_id,)).fetchone()
        if not row:
            raise HTTPException(404)
        item = row_to_dict(row)
        application = {key: item[key] or "" for key in ("organization_name", "organization_type", "registration_number", "darpan_id", "address", "contact_name", "email", "phone")}
        result = run_verification_agents(application, item["extracted"])
        extracted = merge_agent_extraction(item["extracted"], result)
        checks = compare_fields(application, extracted, not bool(extracted.get("extraction_error")))
        report = make_report(application, extracted, checks, not bool(extracted.get("extraction_error")))
        db.execute("UPDATE applications SET extracted = ?, checks = ?, report = ?, agent_status = ?, agent_review = ? WHERE id = ?", (json.dumps(extracted), json.dumps(checks), report, result.get("status", "failed"), json.dumps(result), app_id))
    return RedirectResponse("/admin", status_code=303)


@app.get("/admin/login", response_class=HTMLResponse)
def admin_login(request: Request, error: str = ""):
    return templates.TemplateResponse(request=request, name="login.html", context={"error": error, "configured": bool(os.getenv("ADMIN_TOKEN"))})


@app.post("/admin/login")
def admin_login_submit(token: str = Form(...)):
    expected = os.getenv("ADMIN_TOKEN")
    if not expected:
        return RedirectResponse("/admin/login", status_code=303)
    if not secrets.compare_digest(token, expected):
        return RedirectResponse("/admin/login?error=1", status_code=303)
    response = RedirectResponse("/admin", status_code=303)
    response.set_cookie("admin_token", expected, httponly=True, samesite="strict", secure=os.getenv("COOKIE_SECURE", "0") == "1")
    return response
