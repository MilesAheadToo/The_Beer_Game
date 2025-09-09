from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from fastapi import HTTPException, status
from .. import models
from ..schemas import user as user_schemas
from ..core.security import get_password_hash, verify_password

class UserService:
    def __init__(self, db: Session):
        self.db = db

    def get_user(self, user_id: int) -> models.User:
        """Get a user by ID."""
        user = self.db.query(models.User).filter(models.User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )
        return user

    def get_user_by_email(self, email: str) -> Optional[models.User]:
        """Get a user by email."""
        return self.db.query(models.User).filter(models.User.email == email).first()

    def get_user_by_username(self, username: str) -> Optional[models.User]:
        """Get a user by username."""
        return self.db.query(models.User).filter(models.User.username == username).first()

    def get_users(self, skip: int = 0, limit: int = 100) -> List[models.User]:
        """Get a list of users with pagination."""
        return self.db.query(models.User).offset(skip).limit(limit).all()

    def create_user(self, user: user_schemas.UserCreate) -> models.User:
        """Create a new user."""
        # Check if user with email already exists
        db_user = self.get_user_by_email(user.email)
        if db_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Check if username is taken
        db_user = self.get_user_by_username(user.username)
        if db_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
        
        # Hash the password
        hashed_password = get_password_hash(user.password)
        
        # Create new user
        db_user = models.User(
            username=user.username,
            email=user.email,
            hashed_password=hashed_password,
            full_name=user.full_name,
            is_active=True,
            is_superuser=bool(user.is_superuser)
        )
        
        try:
            self.db.add(db_user)
            self.db.commit()
            self.db.refresh(db_user)
            return db_user
        except SQLAlchemyError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error creating user"
            )

    def update_user(
        self, 
        user_id: int, 
        user_update: user_schemas.UserUpdate,
        current_user: models.User
    ) -> models.User:
        """Update a user."""
        # Only allow users to update their own account unless they're an admin
        if user_id != current_user.id and not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        
        db_user = self.get_user(user_id)
        
        # Update fields if they're provided
        if user_update.email is not None:
            # Check if email is already taken
            existing_user = self.get_user_by_email(user_update.email)
            if existing_user and existing_user.id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            db_user.email = user_update.email
        
        if user_update.full_name is not None:
            db_user.full_name = user_update.full_name
            
        if user_update.is_active is not None and current_user.is_superuser:
            db_user.is_active = user_update.is_active
        if user_update.is_superuser is not None and current_user.is_superuser:
            db_user.is_superuser = bool(user_update.is_superuser)
        
        try:
            self.db.commit()
            self.db.refresh(db_user)
            return db_user
        except SQLAlchemyError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error updating user"
            )

    def delete_user(self, user_id: int, current_user: models.User) -> Dict[str, Any]:
        """Delete a user."""
        # Only allow users to delete their own account unless they're an admin
        if user_id != current_user.id and not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        
        db_user = self.get_user(user_id)
        
        # Prevent deleting the last admin
        if db_user.is_superuser:
            admin_count = self.db.query(models.User).filter(
                models.User.is_superuser == True,
                models.User.is_active == True
            ).count()
            
            if admin_count <= 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot delete the last admin user"
                )
        
        try:
            self.db.delete(db_user)
            self.db.commit()
            return {"message": "User deleted successfully"}
        except SQLAlchemyError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error deleting user"
            )

    def change_password(
        self, 
        user_id: int, 
        current_password: str, 
        new_password: str,
        current_user: models.User
    ) -> Dict[str, str]:
        """Change a user's password."""
        # Only allow users to change their own password unless they're an admin
        if user_id != current_user.id and not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        
        db_user = self.get_user(user_id)
        
        # If it's not an admin changing someone else's password, verify current password
        if not (current_user.is_superuser and user_id != current_user.id):
            if not verify_password(current_password, db_user.hashed_password):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Incorrect current password"
                )
        
        # Update password
        db_user.hashed_password = get_password_hash(new_password)
        
        try:
            self.db.commit()
            return {"message": "Password updated successfully"}
        except SQLAlchemyError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error updating password"
            )
