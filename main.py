import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from groq import Groq
import json

app = FastAPI()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.post("/extract")
async def extract_invoice(request: Request):
    payload = await request.json()
    doc_text = payload.get("text")
    schema = payload.get("schema")

    prompt = f"""
    Analyze the following invoice text and extract the information into a JSON object 
    that strictly follows this schema: {json.dumps(schema)}.
    
    Extraction rules:
    - vendor: Exact proper name.
    - currency: ISO 4217 code (USD, EUR, GBP, INR, JPY).
    - total_amount: Integer only.
    - invoice_date: YYYY-MM-DD.
    - due_in_days: Integer.
    - is_paid: Boolean.
    - priority: low, normal, high, or urgent.
    - contact_email: Lowercased.
    - line_items: Array of objects with sku, quantity, unit_price (int).
    - item_count: Total items as integer.
    
    Text: {doc_text}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": "You are a finance extraction assistant. Return only valid JSON."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)
