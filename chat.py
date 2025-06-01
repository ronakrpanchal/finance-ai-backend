from numpy import rec
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from prompt_schema import ChatPrompt, User
from prompt_utils import prompt_render
import datetime
import os

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def get_user_data(user_id : str):
    client = MongoClient(os.environ.get("MONGO_URI"))
    db = client['finance_ai']
    budgets = db.budgets.find_one({"user_id": user_id})
    user_budget = budgets.get("budget_data",{})
    profiles = db.user_profiles.find_one({"user_id":user_id})
    cash = profiles.get("cash_holdings",0)
    savings = profiles.get("savings",0)
    online = profiles.get("online_holdings",0)
    stocks = profiles.get("stock_investments",0)
    total_savings = profiles.get("total_savings",0)
    currency = profiles.get("currency","")
    return {
        "user_budget": user_budget,
        "cash": cash,
        "savings": savings,
        "online": online,
        "stocks": stocks,
        "total_savings": total_savings,
        "currency": currency
    }
    
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
    user_data = get_user_data(user_id=user_id)
    # print(user_data)
    user = User(data=user_data)
    recent_messages = get_recent_messages(user_id)
    system_prompt = prompt_render(ChatPrompt(user=user,recent_messages=recent_messages))
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