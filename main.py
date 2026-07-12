import os
import re
import json
from fastapi import FastAPI, Request, HTTPException
from groq import Groq, RateLimitError

app = FastAPI()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


@app.post("/extract")
async def extract_invoice(request: Request):
    payload = await request.json()

    doc_text = payload.get("text", "")
    schema = payload.get("schema", {})

    # --------------------------------------------------
    # Extract email directly from the invoice text
    # (prevents LLM from changing characters)
    # --------------------------------------------------
    email_match = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        doc_text,
    )
    extracted_email = (
        email_match.group(0).strip().lower()
        if email_match
        else None
    )

    prompt = f"""
Extract invoice information from the text below.

Return ONLY valid JSON matching this schema:

{json.dumps(schema)}

Rules:

- Do NOT invent values.
- total_amount must be an integer.
- item_count should be 0.
- Extract every line item.
- If a value is missing, return null.

Invoice Text:
\"\"\"
{doc_text}
\"\"\"
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise invoice extraction assistant. "
                        "Return ONLY valid JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Groq rate limit exceeded. Try again later.",
        )

    # --------------------------------------------------
    # Parse JSON
    # --------------------------------------------------
    try:
        data = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Model returned invalid JSON.",
        )

    # Some models wrap the response
    if isinstance(data, dict) and "invoice" in data:
        data = data["invoice"]

    # --------------------------------------------------
    # Override email with regex result
    # --------------------------------------------------
    data["contact_email"] = extracted_email

    # --------------------------------------------------
    # Convert total_amount to integer
    # --------------------------------------------------
    if "total_amount" in data:
        try:
            if isinstance(data["total_amount"], str):
                value = (
                    data["total_amount"]
                    .replace(",", "")
                    .replace("$", "")
                    .replace("₹", "")
                    .strip()
                )
                data["total_amount"] = int(float(value))
            elif data["total_amount"] is not None:
                data["total_amount"] = int(data["total_amount"])
        except Exception:
            data["total_amount"] = None

    # --------------------------------------------------
    # Compute item_count
    # --------------------------------------------------
    if isinstance(data.get("line_items"), list):
        data["item_count"] = len(data["line_items"])
    else:
        data["line_items"] = []
        data["item_count"] = 0

    # --------------------------------------------------
    # Return only required fields
    # --------------------------------------------------
    required_keys = [
        "vendor",
        "currency",
        "total_amount",
        "invoice_date",
        "due_in_days",
        "is_paid",
        "priority",
        "contact_email",
        "line_items",
        "item_count",
    ]

    final_output = {k: data.get(k) for k in required_keys}

    return final_output
