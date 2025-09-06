from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Dict, Any

from app.db.session import SessionLocal

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/health", response_model=Dict[str, str])
async def health_check(db: Session = Depends(get_db)) -> Dict[str, str]:
    """
    Health check endpoint that verifies database connectivity.
    """
    try:
        # Test database connection with text() to avoid deprecation warning
        db.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={"status": "unhealthy", "error": str(e)}
        )
