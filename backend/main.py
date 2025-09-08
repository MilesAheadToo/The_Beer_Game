# /backend/app/main.py
from datetime import datetime, timedelta
import os
from typing import Optional, Dict, Any

from fastapi import (
    FastAPI,
    APIRouter,
    Depends,
    HTTPException,
    status,
    Response,
    Request,
    Cookie,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
API_PREFIX = os.getenv("API_PREFIX", "/api/v1")
FRONTEND_ORIGINS = os.getenv("FRONTEND_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-insecure-secret-key-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

ACCESS_COOKIE_NAME = os.getenv("ACCESS_COOKIE_NAME", "access_token")
REFRESH_COOKIE_NAME = os.getenv("REFRESH_COOKIE_NAME", "refresh_token")

# In dev on http, cookies cannot be "secure=True". Use samesite="lax".
COOKIE_COMMON_KWARGS = dict(httponly=True, samesite="lax", secure=False, path="/")

# ------------------------------------------------------------------------------
# Minimal in-memory user store (replace with your DB/user service)
# ------------------------------------------------------------------------------
# This mirrors your default frontend creds to make dev easy.
_FAKE_USERS = {
    "admin@daybreak.ai": {
        "id": 1,
        "email": "admin@daybreak.ai",
        "name": "Admin",
        "role": "admin",
        "password": "Daybreak@2025",  # DO NOT use plaintext in production.
    }
}

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    refresh_token: Optional[str] = None

class MeResponse(BaseModel):
    id: int
    email: str
    name: str
    role: str

# ------------------------------------------------------------------------------
# JWT utils
# ------------------------------------------------------------------------------
def _create_token(data: Dict[str, Any], expires_delta: timedelta) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_access_token(sub: str, extra: Dict[str, Any]) -> str:
    payload = {"sub": sub, **extra}
    return _create_token(payload, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))

def create_refresh_token(sub: str, extra: Dict[str, Any]) -> str:
    payload = {"sub": sub, "typ": "refresh", **extra}
    return _create_token(payload, timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))

def decode_token(token: str) -> Dict[str, Any]:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

# ------------------------------------------------------------------------------
# Auth helpers
# ------------------------------------------------------------------------------
def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    user = _FAKE_USERS.get(username)
    if not user:
        return None
    if user["password"] != password:
        return None
    return user

def extract_bearer_from_cookie(cookie_val: Optional[str]) -> Optional[str]:
    if not cookie_val:
        return None
    # We accept raw token or "Bearer <token>"
    if cookie_val.lower().startswith("bearer "):
        return cookie_val.split(" ", 1)[1]
    return cookie_val

async def get_current_user(
    request: Request,
    access_cookie: Optional[str] = Cookie(default=None, alias=ACCESS_COOKIE_NAME),
) -> Dict[str, Any]:
    # 1) Try Authorization header first
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    token: Optional[str] = None
    if auth and auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1]
    # 2) Fallback to cookie
    if not token:
        token = extract_bearer_from_cookie(access_cookie)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    try:
        payload = decode_token(token)
        sub = payload.get("sub")
        if not sub:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    # In a real app, fetch user from DB using `sub`
    # Here we map sub back to our fake user by id
    user = None
    for u in _FAKE_USERS.values():
        if str(u["id"]) == str(sub) or u["email"] == sub:
            user = u
            break

    if not user:
        # Token refers to unknown user
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return user

# ------------------------------------------------------------------------------
# App & Middleware
# ------------------------------------------------------------------------------
app = FastAPI(title="Daybreak Beer Game API", version="1.0.0")

# CORS (allow cookies/credentials from the frontend)
origins = [o.strip() for o in FRONTEND_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api = APIRouter(prefix=API_PREFIX, tags=["api"])

# ------------------------------------------------------------------------------
# Health
# ------------------------------------------------------------------------------
@api.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

# Safety alias that some frontends ping
@app.get("/api/health")
def health_alias():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

# ------------------------------------------------------------------------------
# Auth
# ------------------------------------------------------------------------------
@api.post("/auth/login", response_model=TokenResponse, tags=["auth"])
async def login(
    response: Response,
    form: OAuth2PasswordRequestForm = Depends(),
):
    """
    Accepts form-encoded:
      - username
      - password
      - grant_type=password (ignored but accepted for compatibility)
    """
    user = authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    # Use user id as subject
    sub = str(user["id"])
    extra = {"email": user["email"], "role": user["role"]}

    access_token = create_access_token(sub=sub, extra=extra)
    refresh_token = create_refresh_token(sub=sub, extra={"email": user["email"]})

    # Set HttpOnly cookies so the frontend can rely on cookies if desired
    response.set_cookie(
        key=ACCESS_COOKIE_NAME,
        value=f"Bearer {access_token}",
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        **COOKIE_COMMON_KWARGS,
    )
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=refresh_token,
        max_age=REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        **COOKIE_COMMON_KWARGS,
    )

    # Also return tokens in the body to support your current frontend
    return TokenResponse(access_token=access_token, token_type="bearer", refresh_token=refresh_token)

@api.post("/auth/logout", tags=["auth"])
async def logout(response: Response):
    # Clear cookies
    response.delete_cookie(key=ACCESS_COOKIE_NAME, path="/")
    response.delete_cookie(key=REFRESH_COOKIE_NAME, path="/")
    return {"status": "ok"}

@api.get("/auth/me", response_model=MeResponse, tags=["auth"])
async def me(user: Dict[str, Any] = Depends(get_current_user)):
    return MeResponse(id=user["id"], email=user["email"], name=user["name"], role=user["role"])

@api.post("/auth/refresh", response_model=TokenResponse, tags=["auth"])
async def refresh(
    response: Response,
    refresh_cookie: Optional[str] = Cookie(default=None, alias=REFRESH_COOKIE_NAME),
):
    """
    Very minimal refresh implementation:
    - Reads refresh token from cookie
    - Issues a new access token
    """
    if not refresh_cookie:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing refresh token")

    try:
        payload = decode_token(refresh_cookie)
        if payload.get("typ") != "refresh":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")
        sub = payload.get("sub")
        if not sub:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    # Find user
    user = None
    for u in _FAKE_USERS.values():
        if str(u["id"]) == str(sub) or u["email"] == sub:
            user = u
            break
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    new_access = create_access_token(sub=str(user["id"]), extra={"email": user["email"], "role": user["role"]})
    response.set_cookie(
        key=ACCESS_COOKIE_NAME,
        value=f"Bearer {new_access}",
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        **COOKIE_COMMON_KWARGS,
    )
    return TokenResponse(access_token=new_access, token_type="bearer")

# ------------------------------------------------------------------------------
# Example protected route (replace with your real routers)
# ------------------------------------------------------------------------------
@api.get("/secure/ping")
async def secure_ping(user: Dict[str, Any] = Depends(get_current_user)):
    return {"message": f"pong, {user['email']}", "role": user["role"]}

# ------------------------------------------------------------------------------
# Mount router
# ------------------------------------------------------------------------------
app.include_router(api)

# ------------------------------------------------------------------------------
# Root
# ------------------------------------------------------------------------------
@app.get("/")
def root():
    return {"name": "Daybreak Beer Game API", "docs": f"{API_PREFIX}/docs"}

# ------------------------------------------------------------------------------
# Error handlers (optional but useful)
# ------------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
