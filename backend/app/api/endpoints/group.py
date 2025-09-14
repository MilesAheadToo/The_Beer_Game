from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ...db.session import get_db
from ...models import User
from ...schemas.group import Group as GroupSchema, GroupCreate, GroupUpdate
from ...services.group_service import GroupService
from ...core.security import get_current_active_user

router = APIRouter()

def get_group_service(db: Session = Depends(get_db)) -> GroupService:
    return GroupService(db)

@router.get("/", response_model=List[GroupSchema])
def list_groups(group_service: GroupService = Depends(get_group_service), current_user: User = Depends(get_current_active_user)):
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return group_service.get_groups()

@router.post("/", response_model=GroupSchema, status_code=status.HTTP_201_CREATED)
def create_group(group_in: GroupCreate, group_service: GroupService = Depends(get_group_service), current_user: User = Depends(get_current_active_user)):
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return group_service.create_group(group_in)

@router.put("/{group_id}", response_model=GroupSchema)
def update_group(group_id: int, group_update: GroupUpdate, group_service: GroupService = Depends(get_group_service), current_user: User = Depends(get_current_active_user)):
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return group_service.update_group(group_id, group_update)

@router.delete("/{group_id}", response_model=dict)
def delete_group(group_id: int, group_service: GroupService = Depends(get_group_service), current_user: User = Depends(get_current_active_user)):
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return group_service.delete_group(group_id)
