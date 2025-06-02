from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from prompt_schema import ChatPrompt, User
from prompt_utils import prompt_render
import datetime
import pandas as pd
import os
from dateutil.relativedelta import relativedelta

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def get_full_user_profile(user_id: str):
    try:
        # Connect to MongoDB
        client = MongoClient(os.environ.get("MONGO_URI"))
        db = client['finance_ai']

        # ---------- Profile & Budget Info ----------
        budgets = db.budgets.find_one({"user_id": user_id})
        budget_data = budgets.get("budget_data", {
            "income": 0,
            "savings": 0,
            "expenses": []
        }) if budgets else {
            "income": 0,
            "savings": 0,
            "expenses": []
        }

        profiles = db.user_profiles.find_one({"user_id": user_id})
        cash = profiles.get("cash_holdings", 0)
        savings = profiles.get("savings", 0)
        online = profiles.get("online_holdings", 0)
        stocks = profiles.get("stock_investments", 0)
        total_savings = profiles.get("total_savings", 0)
        currency = profiles.get("currency", "")
        # ---------- Transactions, Subscriptions, Debts ----------
        current_month = datetime.datetime.now().strftime('%Y-%m')
        transactions = list(db.transactions.find({
            "user_id": user_id,
            "transaction_date": {"$regex": f"^{current_month}"}
        }))

        subscriptions = list(db.subscriptions.find({"user_id": user_id}))
        debts = list(db.debts.find({"user_id": user_id}))

        df = pd.DataFrame(transactions)
        if df.empty:
            df = pd.DataFrame(columns=["amount", "amount_type", "category", "description"])

        # ---------- Categories ----------
        ESSENTIAL_CATEGORIES = ["Rent", "Utilities", "Healthcare"]
        VARIABLE_CATEGORIES = ["Dining", "Shopping", "Entertainment"]

        # ---------- Calculations ----------
        total_income = df[df["amount_type"] == "credit"]["amount"].sum()
        fixed_expenses = df[df["category"].isin(ESSENTIAL_CATEGORIES)]["amount"].sum()

        variable_expenses = df[df["category"].isin(VARIABLE_CATEGORIES)]
        variable_expenses_summary = variable_expenses.groupby("category")["amount"].sum().to_dict()

        total_subscription_cost = sum(sub.get('cost', 0) for sub in subscriptions)
        total_debt = sum(debt.get('amount', 0) for debt in debts)

        weighted_interest_rate = (
            sum(debt['amount'] * debt.get('interest_rate', 0) for debt in debts) / total_debt
            if total_debt > 0 else 0
        )

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

        # ---------- Final JSON ----------
        user_data_json = {
            "profile_summary": {
                "cash": cash,
                "savings": savings,
                "online": online,
                "stocks": stocks,
                "total_savings": total_savings,
                "currency": currency,
                "budget": budget_data
            },
            "financial_summary": {
                "total_income": total_income,
                "fixed_expenses": fixed_expenses,
                "variable_expenses": variable_expenses_summary,
                "total_subscription_cost": total_subscription_cost,
                "total_debt": total_debt,
                "weighted_interest_rate": weighted_interest_rate,
                "monthly_trends": {
                    "income_trend": calculate_trend(user_id, "credit"),
                    "expense_trend": calculate_trend(user_id, "debit")
                }
            },
            "subscriptions": subscriptions,
            "debts": debts
        }

        return user_data_json

    except Exception as ex:
        print(f"Error during user profile processing: {ex}")
        return None

def get_date_range_last_month_to_today():
    today = datetime.datetime.now()
    first_day_last_month = today.replace(day=1) - relativedelta(months=1)
    return first_day_last_month.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

def get_transactions_between_last_month_and_today(user_id):
    client = MongoClient(os.environ.get("MONGO_URI"))
    db = client['finance_ai']
    start_date, end_date = get_date_range_last_month_to_today()
    
    transactions = list(db.transactions.find({
        "user_id": user_id,
        "transaction_date": {
            "$gte": start_date,
            "$lte": end_date
        }
    }))
    
    return transactions

def store_message(user_id:str,role:str,message:str):
    client = MongoClient(os.environ.get("MONGO_URI"))
    db = client['finance_ai']
    messages_collection = db['chat_memory']
    
    message_data = {
        "user_id": user_id,
        "role": role,
        "message": message,
        "created_at": datetime.datetime.now(datetime.timezone.utc)
    }
    
    messages_collection.insert_one(message_data)
    client.close()    

def get_recent_messages(user_id:str):
    client = MongoClient(os.environ.get("MONGO_URI"))
    db = client['finance_ai']
    messages_collection = db['chat_memory']
    
    recent_messages = list(messages_collection.find({"user_id": user_id}).sort("created_at", -1).limit(10))
    client.close()
    
    return [{"role": msg["role"], "content": msg["message"]} for msg in recent_messages]

def load_model(query:str,user_id:str):
    llm = ChatGroq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        api_key=GROQ_API_KEY
    )
    user_data = get_full_user_profile(user_id=user_id)
    # print(user_data)
    user = User(data=user_data)
    recent_messages = get_recent_messages(user_id)
    recent_expenses = get_transactions_between_last_month_and_today(user_id)
    print(recent_expenses)
    system_prompt = prompt_render(ChatPrompt(user=user,recent_messages=recent_messages,user_expenses=recent_expenses))
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]
    return llm.invoke(messages)


def Chat(query:str,user_id:str) -> str:
    store_message(user_id, "user", query)
    response  = load_model(query,user_id=user_id)
    store_message(user_id, "assistant", response.content)
    return response.content
    
if __name__ == "__main__":
    user_id = "682840c1922dec3aba0733bc"
    query = "just look at my financial data and summarise it for me"
    response = Chat(query,user_id)
    print(response)