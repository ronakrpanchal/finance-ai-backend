from prompt_utils import prompt_render
from prompt_schema import ReceiptPrompt
from pymongo import MongoClient
from groq import Groq
import os
from dotenv import load_dotenv
import json

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
MONGO_URI = os.getenv("MONGO_URI")

def receipt_model(image_url):
    llm = Groq(api_key=GROQ_API_KEY)
    image_prompt = prompt_render(ReceiptPrompt())
    response = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role":"user",
                "content" : [
                    {"type":"text","text":image_prompt},
                    {"type":"image_url","image_url":{"url":image_url}}
                ]
            },
            
        ],
    )
    return response.choices[0].message.content
    
def save_receipt_in_mongodb(user_id, llm_response, date, category):
    client = MongoClient(MONGO_URI)
    db = client['finance_ai']
    collection = db['transactions']
    data = json.loads(llm_response)
    document = []
    for item in data['products']:
        doc = {
            "user_id":user_id,
            "transaction_date":date,
            "amount":item['price'],
            "amount_type":"debit",
            "category":category,
            "description":item['name']
        }
        document.append(doc)
    if document:
        collection.insert_many(document)
        return True
    return False