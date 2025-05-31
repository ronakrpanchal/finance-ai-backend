from typing import Any,Dict
from pydantic import BaseModel

class User(BaseModel):
    data : Dict[str,Any]
    
class ChatPrompt(BaseModel):
    user : User = None
    filename: str = "financial_analyst_prompt.md"