from typing import List, Optional
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
from dotenv import load_dotenv
import os
from pymongo import MongoClient

load_dotenv()

API_KEY = os.environ.get("GROQ_API_KEY")
MODEL_NAME = os.environ.get('MODEL_NAME')
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')

# ---------------------- Pydantic Models ---------------------- #

class BudgetCategory(BaseModel):
    category: str
    allocated_amount: float

class Budget(BaseModel):
    expenses: List[BudgetCategory]

# ---------------------- Model Loader ---------------------- #

def load_model():
    llm = ChatGroq(
        model_name=MODEL_NAME,
        temperature=0.7,
        api_key=API_KEY
    )

    parser = JsonOutputParser(pydantic_object=Budget)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Extract budget details into JSON. 
            Category names must be in Title Case:

            {{
                "expenses": [
                    {{"category": "Category Name", "allocated_amount": amount}}
                ]
            }}
            """),
        ("user", "{input}")
    ])

    chain = prompt | llm | parser
    return chain

# ---------------------- Budget Parser ---------------------- #

def parse_budget(description: str) -> dict:
    chain_ = load_model()
    result = chain_.invoke({"input": description})
    save_json_to_file(result, 'budget_data.json')
    return result

# ---------------------- File Utils ---------------------- #

def save_json_to_file(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=2)

# ---------------------- MongoDB Utils ---------------------- #

def get_mongodb_connection():
    client = MongoClient(MONGO_URI)
    return client

def get_user_budget(user_id):
    client = get_mongodb_connection()
    db = client['finance_ai']
    budgets_collection = db['budgets']
    
    budget = budgets_collection.find_one({'user_id': user_id})
    client.close()
    
    return budget

# ---------------------- Budget Merger ---------------------- #
def merge_budget_data(existing: dict, new: dict) -> dict:
    merged = existing.copy()

    # Merge expenses
    existing_expenses = {item['category']: item for item in merged.get('expenses', [])}
    
    for new_item in new.get('expenses', []):
        cat = new_item['category']
        if cat in existing_expenses:
            existing_expenses[cat]['allocated_amount'] = new_item['allocated_amount']
        else:
            existing_expenses[cat] = new_item

    merged['expenses'] = list(existing_expenses.values())
    return merged

# ---------------------- Save to DB ---------------------- #

def save_in_db(user_id, response):
    client = get_mongodb_connection()
    db = client['finance_ai']
    budgets_collection = db['budgets']

    user_doc = budgets_collection.find_one({'user_id': user_id})

    if user_doc and 'budget_data' in user_doc:
        existing_budget = user_doc['budget_data']
        merged_budget = merge_budget_data(existing_budget, response)
    else:
        merged_budget = response  # No existing budget

    budgets_collection.update_one(
        {'user_id': user_id},
        {'$set': {'budget_data': merged_budget}},
        upsert=True
    )

    save_json_to_file(merged_budget, 'merged_budget.json')  # Optional debug output
    client.close()