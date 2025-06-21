from typing import Any,Dict, List, Optional 
from pydantic import BaseModel

class User(BaseModel):
    data : Dict[str,Any]
    
class ChatPrompt(BaseModel):
    user : User = None
    recent_messages : Optional[List[Dict[str,str]]] = None
    user_expenses : Optional[List] = None
    filename: str = "financial_analyst_prompt.md"
    
class ReceiptPrompt(BaseModel):
    filename : str = "image_prompt.md"