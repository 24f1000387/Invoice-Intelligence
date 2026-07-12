import os
import json
from fastapi import FastAPI, Request
from groq import Groq

app = FastAPI()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


@app.post("/extract")
async def extract_invoice(request: Request):
    payload = await request.json()

    doc_text = payload.get("text", "")
    schema = payload.get("schema", {})

    prompt = f"""
You are an automated financial data extraction system.

Extract information ONLY from the provided invoice text.

Return ONLY valid JSON matching this schema:
{json.dumps(schema)}

EXTRACTION RULES

1. contact_email
- Find the email address in the invoice.
- DO NOT correct spelling.
- DO NOT infer missing letters.
- DO NOT complete domains.
- Copy every character exactly as it appears.
- Return the email in lowercase.

2. total_amount
- Return as an integer.
- Do not include currency symbols or commas.

3. item_count
- Leave as 0.
- It will be calculated automatically.

4. line_items
- Extract every line item present.

5. If a value is missing, return null unless the schema specifies otherwise.

Invoice Text:
\"\"\"
{doc_text}
\"\"\"
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise financial data extraction assistant. "
                    "Return ONLY valid JSON. "
                    "Do not include markdown or explanations."
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

    # Parse JSON
    data = json.loads(response.choices[0].message.content)

    # Some models wrap the output
    if isinstance(data, dict) and "invoice" in data:
        data = data["invoice"]

    # Normalize email (required by evaluator)
    if isinstance(data.get("contact_email"), str):
        data["contact_email"] = (
            data["contact_email"]
            .strip()
            .replace(" ", "")
            .lower()
        )

    # Ensure total_amount is integer
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
            else:
                data["total_amount"] = int(data["total_amount"])
        except Exception:
            data["total_amount"] = None

    # Calculate item_count from line_items
    if isinstance(data.get("line_items"), list):
        data["item_count"] = len(data["line_items"])
    else:
        data["line_items"] = []
        data["item_count"] = 0

    # Return only expected keys
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

    final_output = {key: data.get(key) for key in required_keys}

    return final_output
