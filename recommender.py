import json
import os
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import pandas as pd

# ----------------------- Custom JSON Encoder for MongoDB ObjectId -----------------------
class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# ----------------------- Load Environment -----------------------
load_dotenv()
API_KEY = os.environ.get("GROQ_API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME")
MONGO_URI = os.environ.get("MONGO_URI")

# ----------------------- MongoDB Setup -----------------------
client = MongoClient(MONGO_URI)
db = client['finance_ai']

# ----------------------- Pydantic Output Schema -----------------------
class RiskAssessment(BaseModel):
    debt_risk: str = Field(...)
    savings_risk: str = Field(...)
    subscription_risk: str = Field(...)

class RecommendationOutput(BaseModel):
    recommendations: list[str] = Field(...)
    action_items: list[str] = Field(...)
    risk_assessment: RiskAssessment

# ----------------------- Trend Calculation -----------------------
def calculate_trend(user_id, amount_type):
    pipeline = [
        {"$match": {"user_id": user_id, "amount_type": amount_type}},
        {"$group": {
            "_id": {"$substr": ["$transaction_date", 0, 7]},
            "total": {"$sum": "$amount"}
        }},
        {"$sort": {"_id": -1}},
        {"$limit": 3}
    ]
    results = db.transactions.aggregate(pipeline)
    return [{"month": r["_id"], "total": r["total"]} for r in results]

# ----------------------- Preprocessing -----------------------
def preprocess(user_id):
    try:
        current_month = datetime.now().strftime('%Y-%m')
        transactions = list(db.transactions.find({
            "user_id": user_id,
            "transaction_date": {"$regex": f"^{current_month}"}
        }))

        subscriptions = list(db.subscriptions.find({"user_id": user_id}))
        debts = list(db.debts.find({"user_id": user_id}))
        budget_doc = db.budgets.find_one({"user_id": user_id})

        df = pd.DataFrame(transactions)
        if df.empty:
            df = pd.DataFrame(columns=["amount", "amount_type", "category", "description"])

        ESSENTIAL_CATEGORIES = ["Rent", "Utilities", "Healthcare"]
        VARIABLE_CATEGORIES = ["Dining", "Shopping", "Entertainment"]

        total_income = df[df["amount_type"] == "credit"]["amount"].sum()
        fixed_expenses = df[df["category"].isin(ESSENTIAL_CATEGORIES)]["amount"].sum()

        variable_expenses = df[df["category"].isin(VARIABLE_CATEGORIES)]
        variable_expenses_summary = variable_expenses.groupby("category")["amount"].sum().to_dict()

        total_subscription_cost = sum(sub['cost'] for sub in subscriptions)
        total_debt = sum(debt['amount'] for debt in debts)

        weighted_interest_rate = sum(
            debt['amount'] * debt['interest_rate'] for debt in debts
        ) / total_debt if total_debt > 0 else 0

        budget_data = budget_doc.get("budget_data", {
            "income": 0,
            "savings": 0,
            "expenses": []
        }) if budget_doc else {
            "income": 0,
            "savings": 0,
            "expenses": []
        }

        user_data_json = {
            "financial_summary": {
                "total_income": total_income,
                "fixed_expenses": fixed_expenses,
                "variable_expenses": variable_expenses_summary,
                "total_subscription_cost": total_subscription_cost,
                "total_debt": total_debt,
                "weighted_interest_rate": weighted_interest_rate,
                "budget": {
                    "income": budget_data.get("income", 0),
                    "savings": budget_data.get("savings", 0),
                    "expenses": budget_data.get("expenses", [])
                }
            },
            "subscriptions": subscriptions,
            "debts": debts,
            "monthly_trends": {
                "income_trend": calculate_trend(user_id, "credit"),
                "expense_trend": calculate_trend(user_id, "debit")
            }
        }

        return user_data_json

    except Exception as ex:
        print(f"Error during preprocessing: {ex}")
        return None

# ----------------------- LangChain LLM Chain -----------------------
def load_model():
    llm = ChatGroq(
        model_name=MODEL_NAME,
        temperature=0.7,
        api_key=API_KEY
    )

    parser = JsonOutputParser(pydantic_object=RecommendationOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Analyze the user's financial data and provide comprehensive recommendations.
            Consider their income, expenses, subscriptions, debts, and savings goals.
            Focus on actionable insights and risk assessment.
            
            Data Structure:
            {user_data_json}
            
            Provide recommendations in this format:
            {{
                "recommendations": ["specific recommendation 1", "recommendation 2"],
                "action_items": ["immediate action 1", "action 2"],
                "risk_assessment": {{
                    "debt_risk": "assessment of debt situation",
                    "savings_risk": "assessment of savings progress",
                    "subscription_risk": "assessment of subscription costs"
                }}
            }}"""),
        ("user", "{user_data_json}")
    ])

    return prompt | llm | parser

# ----------------------- Get Recommendations -----------------------
def get_recommendations(user_data_json: dict) -> dict:
    try:
        chain = load_model()
        # Use custom encoder to handle MongoDB objects
        user_data_str = json.dumps(user_data_json, indent=2, cls=MongoJSONEncoder)
        result = chain.invoke({"user_data_json": user_data_str})
        return result
    except Exception as e:
        print(f"Error during model inference: {e}")
        return {"error": f"LLM generation failed: {str(e)}"}

# ----------------------- Main Recommendation Function -----------------------
def financial_recommender(user_id: str):
    user_data_json = preprocess(user_id)
    if not user_data_json:
        return {"error": "Failed to process user data"}
    
    recommendations = get_recommendations(user_data_json)
    return recommendations

# ----------------------- Run Script -----------------------
if __name__ == "__main__":
    user_id = "68245ee0af6dbf213330448c"  # Update accordingly
    recommendations = financial_recommender(user_id)
    print("Financial Recommendations:")
    print(json.dumps(recommendations, indent=2, cls=MongoJSONEncoder))