from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy.future import select
from app.models.user import User

async def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """
    Get a user by email.
    
    Args:
        db: Database session
        email: User's email address
        
    Returns:
        User or None: The user if found, else None
    """
    result = await db.execute(select(User).filter(User.email == email))
    return result.scalars().first()

async def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """
    Get a user by ID.
    
    Args:
        db: Database session
        user_id: User's ID
        
    Returns:
        User or None: The user if found, else None
    """
    result = await db.execute(select(User).filter(User.id == user_id))
    return result.scalars().first()

async def create_user(
    db: Session, 
    email: str, 
    hashed_password: str,
    name: Optional[str] = None,
    roles: Optional[List[str]] = None
) -> User:
    """
    Create a new user.
    
    Args:
        db: Database session
        email: User's email address
        hashed_password: Hashed password
        name: User's name (optional)
        roles: List of user roles (optional)
        
    Returns:
        User: The created user
    """
    user = User(
        email=email,
        hashed_password=hashed_password,
        name=name,
        roles=roles or []
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user

async def update_user(
    db: Session, 
    user: User, 
    **kwargs
) -> User:
    """
    Update a user's attributes.
    
    Args:
        db: Database session
        user: The user to update
        **kwargs: Attributes to update
        
    Returns:
        User: The updated user
    """
    for key, value in kwargs.items():
        if hasattr(user, key):
            setattr(user, key, value)
    
    await db.commit()
    await db.refresh(user)
    return user

async def delete_user(db: Session, user: User) -> bool:
    """
    Delete a user.
    
    Args:
        db: Database session
        user: The user to delete
        
    Returns:
        bool: True if the user was deleted, False otherwise
    """
    await db.delete(user)
    await db.commit()
    return True
