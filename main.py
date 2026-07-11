import os
import json
from fastapi import FastAPI, Request
from groq import Groq

app = FastAPI()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.post("/extract")
async def extract_invoice(request: Request):
    payload = await request.json()
    doc_text = payload.get("text")
    schema = payload.get("schema")

    prompt = f"""
    Analyze the following invoice text. 
    You must return a JSON object that strictly follows this JSON schema: {json.dumps(schema)}.

    CRITICAL RULES:
    - vendor: Exact proper name.
    - currency: ISO 4217 code.
    - total_amount: Integer.
    - invoice_date: YYYY-MM-DD.
    - due_in_days: Integer.
    - is_paid: Boolean.
    - priority: low, normal, high, or urgent.
    - contact_email: Lowercased, EXTRACT VERBATIM. Do not correct spelling, 
      do not abbreviate, do not assume common domain names. Copy the characters exactly as they appear in the text.
    - line_items: Array of objects with sku, quantity, unit_price (int).
    - item_count: DO NOT EXTRACT. I will calculate this myself.

    Text: {doc_text}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a precise financial data extraction assistant. Return ONLY valid JSON matching the provided schema."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    # Parse and cleanup
    data = json.loads(response.choices[0].message.content)
    
    # If LLM wrapped it, extract inner content
    if "invoice" in data:
        data = data["invoice"]

    # FORCE item_count to be accurate based on the array length
    if "line_items" in data and isinstance(data["line_items"], list):
        data["item_count"] = int(len(data["line_items"]))
    else:
        data["item_count"] = 0

    # Ensure only the exact keys required are returned (Filter out hallucinations)
    required_keys = [
        "vendor", "currency", "total_amount", "invoice_date", 
        "due_in_days", "is_paid", "priority", "contact_email", 
        "line_items", "item_count"
    ]
    
    final_output = {k: data.get(k) for k in required_keys}
    
    return final_output
    
