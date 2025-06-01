from typing import Any,Dict, List, Optional
from pydantic import BaseModel

class User(BaseModel):
    data : Dict[str,Any]
    
class ChatPrompt(BaseModel):
    user : User = None
    recent_messages : Optional[List[Dict[str,str]]] = None
    filename: str = "financial_analyst_prompt.md"