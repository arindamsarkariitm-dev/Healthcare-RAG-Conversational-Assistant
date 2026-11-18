from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
VSTORE_DIR = BASE_DIR / "vector_store"

PDF_DOCS = {
    "patient_education_hypertension.pdf": [
        "Title: Hypertension Patient Education Guide",
        "Source: CDC public health resources",
        "Hypertension is high blood pressure, which can raise the risk of heart disease, stroke, and kidney problems.",
        "Management recommendations:",
        "- Check blood pressure regularly.",
        "- Follow a heart-healthy diet with fruits, vegetables, and lean protein.",
        "- Limit sodium intake and exercise at least 150 minutes weekly.",
        "- Take medications exactly as prescribed.",
        "Seek care if blood pressure remains elevated or if there is chest pain, shortness of breath, or sudden vision changes.",
    ],
    "patient_education_diabetes.pdf": [
        "Title: Diabetes Patient Education Guide",
        "Source: WHO public health materials",
        "Diabetes is a chronic condition characterized by elevated blood sugar due to insufficient insulin production or insulin resistance.",
        "Types of diabetes:",
        "- Type 1: The body does not produce insulin.",
        "- Type 2: The body does not use insulin properly.",
        "- Gestational diabetes: Develops during pregnancy.",
        "Management includes monitoring blood glucose, healthy eating, regular exercise, and taking medications as prescribed.",
    ],
    "patient_education_asthma.pdf": [
        "Title: Asthma Patient Education Guide",
        "Source: National Health Service public documents",
        "Asthma is a chronic respiratory condition that causes airway inflammation and breathing difficulty.",
        "Common asthma triggers include smoke, pollen, cold air, and exercise.",
        "Management strategies:",
        "- Use inhalers and medications as prescribed.",
        "- Monitor symptoms and avoid known triggers.",
        "- Develop an asthma action plan with your healthcare provider.",
    ],
    "hospital_faq_visiting_hours.pdf": [
        "Title: Hospital FAQ - Visiting Hours",
        "Source: Hospital public information portal",
        "Visiting hours are daily from 9:00 AM to 8:00 PM.",
        "Guidelines:",
        "- Check in at the front desk and follow safety guidelines.",
        "- Up to two visitors are allowed at one time unless otherwise directed.",
        "- Certain units, such as intensive care, may have special restrictions.",
        "Exceptions can be made for critical patients upon care team approval.",
    ],
    "hospital_faq_insurance_coverage.pdf": [
        "Title: Hospital FAQ - Insurance Coverage",
        "Source: Government healthcare portal",
        "Patients should verify insurance benefits before receiving hospital services.",
        "Common questions:",
        "- Does my insurance cover emergency services?",
        "- Are specialists and diagnostic tests included?",
        "- Contact your insurer for coverage details and pre-authorization requirements.",
        "If coverage is uncertain, speak with hospital billing staff before treatment.",
    ],
}

PDF_TEMPLATE = """%PDF-1.4\n{objects}\nxref\n0 {count}\n0000000000 65535 f \n{xref_entries}\ntrailer << /Size {count} /Root 1 0 R >>\nstartxref\n{startxref}\n%%EOF\n"""


def escape_pdf_text(line: str) -> str:
    return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(lines):
    text_lines = [escape_pdf_text(line) for line in lines]
    content = "BT /F1 12 Tf 50 760 Td (" + text_lines[0] + ") Tj\n"
    y = 740
    for line in text_lines[1:]:
        content += f"0 -18 Td ({line}) Tj\n"
    content += "ET\n"
    content_bytes = content.encode("utf-8")
    objs = []
    objs.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objs.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    objs.append(b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n")
    objs.append(f"4 0 obj << /Length {len(content_bytes)} >> stream\n".encode("utf-8") + content_bytes + b"endstream endobj\n")
    objs.append(b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")

    body = b"".join(objs)
    xref_entries = []
    offset = len("%PDF-1.4\n")
    for obj in objs:
        xref_entries.append(f"{offset:010d} 00000 n \n")
        offset += len(obj)
    final = PDF_TEMPLATE.format(
        objects=body.decode("latin1"),
        count=len(objs) + 1,
        xref_entries="".join(xref_entries),
        startxref=offset,
    )
    return final.encode("latin1")


if __name__ == "__main__":
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for path in DOCS_DIR.glob("*.md"):
        path.unlink()
    for filename, lines in PDF_DOCS.items():
        pdf_path = DOCS_DIR / filename
        pdf_bytes = build_pdf(lines)
        pdf_path.write_bytes(pdf_bytes)
    VSTORE_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / ".env").write_text("OPENAI_API_KEY=\nNew_OPENAI_API_KEY=\n", encoding="utf-8")
    print("PDF documents created, vector_store directory ensured, and .env file prepared.")
