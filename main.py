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
    You are an automated data entry clerk. Your task is to extract information exactly as written in the invoice.
    
    JSON Schema to follow: {json.dumps(schema)}

    EXTRACTION RULES:
    1. For 'contact_email', DO NOT USE your knowledge of words or common domains. 
       Look at the text provided, find the email address, and copy it character-by-character. 
       If the email is "john.khan@bluewaveanalyt.co", you MUST write "john.khan@bluewaveanalyt.co".
       Do not add, remove, or change any characters.
    2. 'total_amount': Extract as integer.
    3. 'item_count': Leave blank or 0, I will calculate it.
    4. Follow the provided schema strictly.

    Text:
    \"\"\"
    {doc_text}
    \"\"\"
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
    
