from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import Dict, Any

from app.db.session import get_db

router = APIRouter()

@router.get("/health", response_model=Dict[str, str])
async def health_check(db: AsyncSession = Depends(get_db)) -> Dict[str, str]:
    """
    Health check endpoint that verifies database connectivity.
    """
    try:
        # Test database connection with text() to avoid deprecation warning
        result = await db.execute(text("SELECT 1"))
        await result.scalar()  # Ensure the query executes
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={"status": "unhealthy", "error": str(e)}
        )
