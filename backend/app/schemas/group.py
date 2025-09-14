from typing import Optional, List
from pydantic import BaseModel

from .user import UserCreate, User

class GroupBase(BaseModel):
    name: str
    description: Optional[str] = None
    logo: Optional[str] = None

class GroupCreate(GroupBase):
    admin: UserCreate

class GroupUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    logo: Optional[str] = None

class Group(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    logo: Optional[str] = None
    admin: User

    class Config:
        from_attributes = True
