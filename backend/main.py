# /backend/app/main.py
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import json
import logging
import os
import threading
import time
from typing import Optional, Dict, Any, List, Set, Literal

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
from pydantic import BaseModel, Field

from sqlalchemy import or_
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.services.group_service import GroupService
from app.services.bootstrap import build_default_group_payload, ensure_default_group_and_game
from app.schemas.group import GroupCreate, GroupUpdate, Group as GroupSchema
from app.schemas.game import GameCreate, PricingConfig, NodePolicy, DemandPattern
from app.schemas.player import PlayerAssignment, PlayerType as PlayerTypeSchema
from app.db.session import sync_engine
from app.models.game import Game as DbGame, GameStatus as DbGameStatus, Round, PlayerAction
from app.models.player import Player, PlayerRole, PlayerStrategy, PlayerType as PlayerModelType
from app.models.supply_chain_config import SupplyChainConfig
from app.models.supply_chain import (
    PlayerInventory,
    Order as SupplyOrder,
    GameRound as SupplyGameRound,
    PlayerRound as SupplyPlayerRound,
)
from app.models.user import User, UserTypeEnum
from app.core.security import verify_password
from app.services.agents import AgentManager, AgentType, AgentStrategy as AgentStrategyEnum
from app.services.mixed_game_service import MixedGameService

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
API_PREFIX = os.getenv("API_PREFIX", "/api/v1")
# Allow common local frontend ports so CORS works out of the box
FRONTEND_ORIGINS = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,"
    "http://localhost:8080,http://127.0.0.1:8080",
)
SECRET_KEY = os.getenv("SECRET_KEY", "dev-insecure-secret-key-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

ACCESS_COOKIE_NAME = os.getenv("ACCESS_COOKIE_NAME", "access_token")
REFRESH_COOKIE_NAME = os.getenv("REFRESH_COOKIE_NAME", "refresh_token")

# In dev on http, cookies cannot be "secure=True". Use samesite="lax".
COOKIE_COMMON_KWARGS = dict(httponly=True, samesite="lax", secure=False, path="/")

# Filesystem roots used for derived training metadata
BACKEND_ROOT = Path(__file__).resolve().parent
CHECKPOINT_ROOT = BACKEND_ROOT / "checkpoints" / "supply_chain_configs"

# ------------------------------------------------------------------------------
# Minimal in-memory user store (replace with your DB/user service)
# ------------------------------------------------------------------------------
# This mirrors your default frontend creds to make dev easy.
_FAKE_USERS = {
    # Keyed by email for canonical lookup
    "systemadmin@daybreak.ai": {
        "id": 1,
        "email": "systemadmin@daybreak.ai",
        "name": "System Admin",
        "role": "systemadmin",
        # Dev-only password to simplify getting started
        "passwords": {"Daybreak@2025", "Daybreak@2025!"},
        "aliases": {"systemadmin", "superadmin"},
        "group_id": None,
        "is_superuser": True,
        "user_type": "system_admin",
    }
}

# Allow default group admin to sign in using the same lightweight auth shim.
_FAKE_USERS["groupadmin@daybreak.ai"] = {
    "id": 7,
    "email": "groupadmin@daybreak.ai",
    "name": "Group Administrator",
    "role": "groupadmin",
    "passwords": {"Daybreak@2025", "DayBreak@2025"},
    "aliases": {"groupadmin", "defaultadmin"},
    "group_id": 1,
    "is_superuser": False,
    "user_type": "group_admin",
}

# Logger used across helpers/routes
logger = logging.getLogger(__name__)

# Directory for optional debug logs written during game execution
DEBUG_LOG_DIR = Path(
    os.getenv("BEER_GAME_DEBUG_DIR")
    or (Path(__file__).resolve().parent / "debug_logs")
)

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
    group_id: Optional[int] = None
    user_type: Optional[str] = None
    is_superuser: bool = False


class OrderSubmission(BaseModel):
    player_id: int
    quantity: int
    comment: Optional[str] = None


class GameUpdatePayload(BaseModel):
    name: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    is_public: Optional[bool] = None
    max_rounds: Optional[int] = Field(None, ge=1, le=1000)
    progression_mode: Optional[Literal["supervised", "unsupervised"]] = None
    demand_pattern: Optional[DemandPattern] = None
    pricing_config: Optional[PricingConfig] = None
    node_policies: Optional[Dict[str, NodePolicy]] = None
    system_config: Optional[Dict[str, Any]] = None
    global_policy: Optional[Dict[str, Any]] = None
    player_assignments: Optional[List[PlayerAssignment]] = None

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
    """Authenticate against the database first, falling back to the in-memory dev users."""

    lookup = (username or "").strip()
    if not lookup:
        return None

    # Map common aliases (e.g. "systemadmin") onto their canonical emails
    alias_match = None
    if "@" not in lookup:
        token = lookup.lower()
        for candidate in _FAKE_USERS.values():
            aliases = {a.lower() for a in candidate.get("aliases", set())}
            if token in aliases:
                alias_match = candidate["email"]
                break
    canonical = alias_match or lookup

    # Primary path: validate against persisted users so we pick up real group assignments
    session: Optional[Session] = None
    try:
        session = SessionLocal()
        db_user = session.query(User).filter(
            or_(User.email == canonical, User.username == canonical)
        ).first()
        if db_user and getattr(db_user, "hashed_password", None):
            try:
                if verify_password(password, db_user.hashed_password):
                    return _build_user_payload_from_model(db_user)
            except Exception:
                # If password verification fails unexpectedly, fall back to dev users
                pass
    finally:
        if session is not None:
            session.close()

    # Development fallback: use in-memory credentials
    user = _FAKE_USERS.get(canonical) or _FAKE_USERS.get(lookup)
    if not user:
        token = lookup.lower()
        for candidate in _FAKE_USERS.values():
            if token in {a.lower() for a in candidate.get("aliases", set())}:
                user = candidate
                break
    if not user:
        return None
    if password not in user.get("passwords", set()):
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

    user: Optional[Dict[str, Any]] = None
    session: Optional[Session] = None
    try:
        session = SessionLocal()
        db_user: Optional[User] = None
        if str(sub).isdigit():
            db_user = session.get(User, int(sub))
        if db_user is None:
            email_hint = payload.get("email")
            if email_hint:
                db_user = session.query(User).filter(User.email == email_hint).first()
        if db_user is None and sub:
            db_user = session.query(User).filter(User.email == str(sub)).first()
        if db_user is not None:
            user = _build_user_payload_from_model(db_user)
    finally:
        if session is not None:
            session.close()

    if user is None:
        # Fallback to the in-memory dev users if we couldn't resolve via the database
        for u in _FAKE_USERS.values():
            if str(u["id"]) == str(sub) or u["email"].lower() == str(sub).lower():
                user = u
                break

    if user is None:
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
    display_name = user.get("name") or user.get("full_name") or user.get("username") or user["email"]
    return MeResponse(
        id=user["id"],
        email=user["email"],
        name=display_name,
        role=user.get("role", "player"),
        group_id=user.get("group_id"),
        user_type=user.get("user_type"),
        is_superuser=bool(user.get("is_superuser", False)),
    )

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

# Alias to match frontend expectation '/auth/refresh-token'
@api.post("/auth/refresh-token", response_model=TokenResponse, tags=["auth"])
async def refresh_alias(response: Response, refresh_cookie: Optional[str] = Cookie(default=None, alias=REFRESH_COOKIE_NAME)):
    return await refresh(response=response, refresh_cookie=refresh_cookie)

# ------------------------------------------------------------------------------
# Lightweight user & supply chain helpers for the dev backend
# ------------------------------------------------------------------------------

@api.get("/users")
async def list_users(
    limit: int = 250,
    offset: int = 0,
    user_type: Optional[str] = None,
    group_id: Optional[int] = None,
    search: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    db = SyncSessionLocal()
    try:
        query = db.query(User).order_by(User.created_at.desc())

        if not _is_system_admin_user(current_user):
            group_filter = _extract_group_id(current_user)
            if not group_filter:
                return []
            query = query.filter(User.group_id == group_filter)
        elif group_id is not None:
            query = query.filter(User.group_id == group_id)

        if user_type:
            normalized = user_type.strip().upper()
            try:
                enum_value = UserTypeEnum[normalized]
                query = query.filter(User.user_type == enum_value.value)
            except KeyError:
                query = query.filter(User.user_type == user_type)

        if search:
            token = f"%{search.strip()}%"
            query = query.filter(
                or_(
                    User.email.ilike(token),
                    User.username.ilike(token),
                    User.full_name.ilike(token),
                )
            )

        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)

        rows = query.all()
        return [_serialize_user_record(row) for row in rows]
    finally:
        db.close()


@api.get("/supply-chain-config/")
async def list_supply_chain_configs(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    db = SyncSessionLocal()
    try:
        query = db.query(SupplyChainConfig).order_by(SupplyChainConfig.created_at.desc())
        if not _is_system_admin_user(current_user):
            group_filter = _extract_group_id(current_user)
            if not group_filter:
                return []
            query = query.filter(SupplyChainConfig.group_id == group_filter)

        configs = query.all()
        return [_serialize_supply_chain_config(cfg) for cfg in configs]
    finally:
        db.close()

# CSRF token endpoint to satisfy frontend interceptor
@api.get("/auth/csrf-token", tags=["auth"])
async def csrf_token(response: Response):
    import secrets
    token = secrets.token_urlsafe(32)
    # Set a non-HttpOnly cookie accessible to JS for header echo
    response.set_cookie(
        key="csrf_token",
        value=token,
        max_age=7 * 24 * 60 * 60,
        httponly=False,
        samesite="lax",
        secure=False,
        path="/",
    )
    return {"csrf_token": token}

# ------------------------------------------------------------------------------
# Example protected route (replace with your real routers)
# ------------------------------------------------------------------------------
@api.get("/secure/ping")
async def secure_ping(user: Dict[str, Any] = Depends(get_current_user)):
    return {"message": f"pong, {user['email']}", "role": user["role"]}

# ------------------------------------------------------------------------------
# System config (master ranges) and Model config (game model setup)
# ------------------------------------------------------------------------------
from pydantic import Field, validator
from typing import List, Optional, Mapping
import json
from sqlalchemy import create_engine, Column, Integer, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
SYSTEM_CONFIG_PATH = os.path.join(DATA_DIR, "system_config.json")
MODEL_CONFIG_PATH = os.path.join(DATA_DIR, "model_config.json")
os.makedirs(DATA_DIR, exist_ok=True)

# Build SQLAlchemy engine from env (MariaDB/MySQL preferred; fallback to SQLite for dev)
def _build_engine():
    # Highest priority: DATABASE_URL
    database_url = os.getenv("DATABASE_URL", "").strip()
    if database_url:
        return create_engine(database_url)

    # Next: MariaDB/MYSQL discrete env vars
    host = os.getenv("MARIADB_HOST") or os.getenv("MYSQL_SERVER") or os.getenv("MYSQL_HOST") or "localhost"
    port = int(os.getenv("MARIADB_PORT") or os.getenv("MYSQL_PORT") or 3306)
    name = os.getenv("MARIADB_DATABASE") or os.getenv("MYSQL_DB") or os.getenv("MYSQL_DATABASE") or "beer_game"
    user = os.getenv("MARIADB_USER") or os.getenv("MYSQL_USER") or "root"
    pwd = os.getenv("MARIADB_PASSWORD") or os.getenv("MYSQL_PASSWORD") or ""

    mariadb_url = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{name}?charset=utf8mb4"
    try:
        eng = create_engine(mariadb_url, pool_pre_ping=True)
        # Attempt a lightweight connection test
        with eng.connect() as conn:
            conn.execute("SELECT 1")
        return eng
    except Exception:
        # Fallback to SQLite (dev only)
        db_path = os.path.join(DATA_DIR, "app.db")
        return create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()
SyncSessionLocal = sessionmaker(bind=sync_engine, autocommit=False, autoflush=False)


def get_sync_session():
    db = SyncSessionLocal()
    try:
        yield db
    finally:
        db.close()


def require_system_admin(user: Dict[str, Any]):
    role = (user.get("role") or "").lower()
    if role not in {"systemadmin", "superadmin"}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")


def _default_group_payload() -> GroupCreate:
    return build_default_group_payload()


def _serialize_user_record(user: User) -> Dict[str, Any]:
    if hasattr(user.user_type, "value"):
        user_type = user.user_type.value
    else:
        user_type = str(user.user_type) if user.user_type is not None else None

    return {
        "id": user.id,
        "email": user.email,
        "username": user.username,
        "full_name": getattr(user, "full_name", None),
        "group_id": user.group_id,
        "user_type": user_type,
        "is_active": bool(getattr(user, "is_active", True)),
        "is_superuser": bool(getattr(user, "is_superuser", False)),
        "created_at": _iso(getattr(user, "created_at", None)),
        "updated_at": _iso(getattr(user, "updated_at", None)),
        "last_login": _iso(getattr(user, "last_login", None)),
    }


def _normalize_role_from_user(user_type: Optional[str], is_superuser: bool) -> str:
    token = (user_type or "").strip().lower()
    if is_superuser or token in {"system_admin", "systemadmin", "superadmin"}:
        return "systemadmin"
    if token in {"group_admin", "groupadmin"}:
        return "groupadmin"
    return "player"


def _build_user_payload_from_model(user: User) -> Dict[str, Any]:
    data = _serialize_user_record(user)
    name = data.get("full_name") or data.get("username") or data.get("email")
    user_type = (data.get("user_type") or "")
    role = _normalize_role_from_user(user_type, bool(data.get("is_superuser")))
    payload = {
        "id": data["id"],
        "email": data["email"],
        "name": name,
        "role": role,
        "group_id": data.get("group_id"),
        "is_superuser": bool(data.get("is_superuser")),
        "user_type": user_type,
    }
    return payload


def _supply_chain_checkpoint_path(config_id: int) -> Path:
    return CHECKPOINT_ROOT / f"config_{config_id}" / "temporal_gnn.pt"


def _serialize_supply_chain_config(cfg: SupplyChainConfig) -> Dict[str, Any]:
    model_path = cfg.trained_model_path or None
    derived_path: Optional[str] = None
    if model_path and Path(model_path).exists():
        derived_path = model_path
    else:
        candidate = _supply_chain_checkpoint_path(cfg.id)
        if candidate.exists():
            derived_path = str(candidate)

    training_status = (cfg.training_status or "").strip()
    normalized_status = training_status.lower()
    needs_training = bool(cfg.needs_training)
    trained_at_iso = _iso(getattr(cfg, "trained_at", None))

    if derived_path and normalized_status in {"", "pending", "in_progress", "needs_training"}:
        training_status = "trained"
    if derived_path and needs_training:
        needs_training = False
    if derived_path and not trained_at_iso:
        try:
            trained_at_iso = datetime.utcfromtimestamp(Path(derived_path).stat().st_mtime).isoformat() + "Z"
        except OSError:
            trained_at_iso = None

    return {
        "id": cfg.id,
        "name": cfg.name,
        "description": cfg.description,
        "is_active": bool(cfg.is_active),
        "group_id": cfg.group_id,
        "created_at": _iso(getattr(cfg, "created_at", None)),
        "updated_at": _iso(getattr(cfg, "updated_at", None)),
        "needs_training": needs_training,
        "training_status": training_status or ("trained" if derived_path else "pending"),
        "trained_at": trained_at_iso,
        "trained_model_path": derived_path,
    }


class SystemConfigRow(Base):
    __tablename__ = "system_config"
    id = Column(Integer, primary_key=True)
    version = Column(Integer, default=1)
    payload = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class ModelConfigRow(Base):
    __tablename__ = "model_config"
    id = Column(Integer, primary_key=True)
    version = Column(Integer, default=1)
    payload = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


try:
    Base.metadata.create_all(bind=engine)
except Exception:
    pass


class Range(BaseModel):
    min: float
    max: float


class SystemConfigModel(BaseModel):
    name: str = Field(default="Default System Config")
    info_delay: Range = Field(default=Range(min=0, max=8))
    ship_delay: Range = Field(default=Range(min=0, max=8))
    init_inventory: Range = Field(default=Range(min=0, max=1000))
    holding_cost: Range = Field(default=Range(min=0, max=100))
    backlog_cost: Range = Field(default=Range(min=0, max=200))
    max_inbound_per_link: Range = Field(default=Range(min=10, max=2000))
    max_order: Range = Field(default=Range(min=10, max=2000))
    price: Range = Field(default=Range(min=0, max=10000))
    standard_cost: Range = Field(default=Range(min=0, max=10000))
    min_order_qty: Range = Field(default=Range(min=0, max=1000))


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _read_system_cfg() -> SystemConfigModel:
    # Try DB first
    try:
        db = SessionLocal()
        row = db.query(SystemConfigRow).order_by(SystemConfigRow.updated_at.desc()).first()
        if row and row.payload:
            return SystemConfigModel(**json.loads(row.payload))
    except Exception:
        pass
    finally:
        try:
            db.close()
        except Exception:
            pass
    # Fallback: read file and seed DB
    try:
        if os.path.exists(SYSTEM_CONFIG_PATH):
            with open(SYSTEM_CONFIG_PATH, "r") as f:
                data = json.load(f)
                try:
                    db = SessionLocal()
                    seed = SystemConfigRow(id=1, version=1, payload=json.dumps(data), created_at=datetime.utcnow(), updated_at=datetime.utcnow())
                    db.add(seed)
                    db.commit()
                except Exception:
                    pass
                finally:
                    try:
                        db.close()
                    except Exception:
                        pass
                return SystemConfigModel(**data)
    except Exception:
        pass
    return SystemConfigModel()


@api.get("/config/system", response_model=SystemConfigModel)
def get_system_config():
    return _read_system_cfg()


@api.put("/config/system", response_model=SystemConfigModel)
def put_system_config(cfg: SystemConfigModel):
    # Save to DB (single row upsert)
    try:
        payload = json.dumps(cfg.dict())
        db = SessionLocal()
        row = db.query(SystemConfigRow).filter(SystemConfigRow.id == 1).first()
        now = datetime.utcnow()
        if row:
            row.payload = payload
            row.updated_at = now
        else:
            row = SystemConfigRow(id=1, version=1, payload=payload, created_at=now, updated_at=now)
            db.add(row)
        db.commit()
        return cfg
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save system config: {e}")
    finally:
        try:
            db.close()
        except Exception:
            pass


# ---------------------- Model Config ----------------------
class Item(BaseModel):
    id: str
    name: str


class Site(BaseModel):
    id: str
    type: str  # manufacturer|distributor|wholesaler|retailer
    name: str
    items_sold: List[str] = Field(default_factory=list)


class SiteItemSettings(BaseModel):
    inventory_target: float
    holding_cost: float
    backorder_cost: float
    avg_selling_price: float
    standard_cost: float
    moq: float


class Lane(BaseModel):
    from_site_id: str
    to_site_id: str
    item_id: str
    lead_time: float
    capacity: Optional[float] = None
    otif_target: Optional[float] = Field(default=None, description="0-1 fraction or 0-100 percent")

    @validator("otif_target")
    def _normalize_otif(cls, v):
        if v is None:
            return v
        # Accept 0-1 or 0-100; normalize to 0-1
        return v / 100.0 if v > 1 else v


class RetailerDemand(BaseModel):
    distribution: str = Field(default="profile")  # profile|poisson|normal
    params: Mapping[str, float] = Field(default_factory=dict)
    expected_delivery_offset: Optional[float] = 0.0


class ModelConfig(BaseModel):
    version: int = 1
    items: List[Item]
    sites: List[Site]
    # site_item_settings[siteId][itemId] = SiteItemSettings
    site_item_settings: Mapping[str, Mapping[str, SiteItemSettings]]
    lanes: List[Lane]
    retailer_demand: RetailerDemand
    manufacturer_lead_times: Mapping[str, float] = Field(default_factory=dict)


def _read_model_cfg() -> ModelConfig:
    # Try DB first
    try:
        db = SessionLocal()
        row = db.query(ModelConfigRow).order_by(ModelConfigRow.updated_at.desc()).first()
        if row and row.payload:
            return ModelConfig(**json.loads(row.payload))
    except Exception:
        pass
    finally:
        try:
            db.close()
        except Exception:
            pass
    # Fallback to file and seed DB
    if os.path.exists(MODEL_CONFIG_PATH):
        with open(MODEL_CONFIG_PATH, "r") as f:
            data = json.load(f)
            try:
                db = SessionLocal()
                seed = ModelConfigRow(id=1, version=1, payload=json.dumps(data), created_at=datetime.utcnow(), updated_at=datetime.utcnow())
                db.add(seed)
                db.commit()
            except Exception:
                pass
            finally:
                try:
                    db.close()
                except Exception:
                    pass
            return ModelConfig(**data)
    # Default classic beer game
    default = ModelConfig(
        items=[Item(id="item_1", name="Item 1")],
        sites=[
            Site(id="manufacturer_1", type="manufacturer", name="Manufacturer 1", items_sold=["item_1"]),
            Site(id="distributor_1", type="distributor", name="Distributor 1", items_sold=["item_1"]),
            Site(id="wholesaler_1", type="wholesaler", name="Wholesaler 1", items_sold=["item_1"]),
            Site(id="retailer_1", type="retailer", name="Retailer 1", items_sold=["item_1"]),
        ],
        site_item_settings={
            "manufacturer_1": {"item_1": SiteItemSettings(inventory_target=20, holding_cost=0.5, backorder_cost=1.0, avg_selling_price=7.0, standard_cost=5.0, moq=0)},
            "distributor_1": {"item_1": SiteItemSettings(inventory_target=20, holding_cost=0.5, backorder_cost=1.0, avg_selling_price=7.0, standard_cost=5.0, moq=0)},
            "wholesaler_1": {"item_1": SiteItemSettings(inventory_target=20, holding_cost=0.5, backorder_cost=1.0, avg_selling_price=7.0, standard_cost=5.0, moq=0)},
            "retailer_1": {"item_1": SiteItemSettings(inventory_target=20, holding_cost=0.5, backorder_cost=1.0, avg_selling_price=7.0, standard_cost=5.0, moq=0)},
        },
        lanes=[
            Lane(from_site_id="manufacturer_1", to_site_id="distributor_1", item_id="item_1", lead_time=2, capacity=None, otif_target=0.95),
            Lane(from_site_id="distributor_1", to_site_id="wholesaler_1", item_id="item_1", lead_time=2, capacity=None, otif_target=0.95),
            Lane(from_site_id="wholesaler_1", to_site_id="retailer_1", item_id="item_1", lead_time=2, capacity=None, otif_target=0.95),
        ],
        retailer_demand=RetailerDemand(distribution="profile", params={"week1_4": 4, "week5_plus": 8}, expected_delivery_offset=1),
        manufacturer_lead_times={"item_1": 2},
    )
    return default


def _validate_model_config(cfg: ModelConfig, ranges: SystemConfigModel):
    # Validate site-item settings against ranges
    errors = []
    for site_id, item_map in cfg.site_item_settings.items():
        for item_id, s in item_map.items():
            if not (ranges.init_inventory.min <= s.inventory_target <= ranges.init_inventory.max):
                errors.append(f"site {site_id} item {item_id}: inventory_target {s.inventory_target} not in [{ranges.init_inventory.min},{ranges.init_inventory.max}]")
            if not (ranges.holding_cost.min <= s.holding_cost <= ranges.holding_cost.max):
                errors.append(f"site {site_id} item {item_id}: holding_cost {s.holding_cost} not in [{ranges.holding_cost.min},{ranges.holding_cost.max}]")
            if not (ranges.backlog_cost.min <= s.backorder_cost <= ranges.backlog_cost.max):
                errors.append(f"site {site_id} item {item_id}: backorder_cost {s.backorder_cost} not in [{ranges.backlog_cost.min},{ranges.backlog_cost.max}]")
            if not (ranges.price.min <= s.avg_selling_price <= ranges.price.max):
                errors.append(f"site {site_id} item {item_id}: avg_selling_price {s.avg_selling_price} not in [{ranges.price.min},{ranges.price.max}]")
            if not (ranges.standard_cost.min <= s.standard_cost <= ranges.standard_cost.max):
                errors.append(f"site {site_id} item {item_id}: standard_cost {s.standard_cost} not in [{ranges.standard_cost.min},{ranges.standard_cost.max}]")
            if not (ranges.min_order_qty.min <= s.moq <= ranges.min_order_qty.max):
                errors.append(f"site {site_id} item {item_id}: moq {s.moq} not in [{ranges.min_order_qty.min},{ranges.min_order_qty.max}]")

    # Validate lanes
    for lane in cfg.lanes:
        if not (ranges.ship_delay.min <= lane.lead_time <= ranges.ship_delay.max):
            errors.append(f"lane {lane.from_site_id}->{lane.to_site_id} item {lane.item_id}: lead_time {lane.lead_time} not in [{ranges.ship_delay.min},{ranges.ship_delay.max}]")
        if lane.capacity is not None and not (ranges.max_inbound_per_link.min <= lane.capacity <= ranges.max_inbound_per_link.max):
            errors.append(f"lane {lane.from_site_id}->{lane.to_site_id} item {lane.item_id}: capacity {lane.capacity} not in [{ranges.max_inbound_per_link.min},{ranges.max_inbound_per_link.max}]")

    if errors:
        raise HTTPException(status_code=422, detail={"message": "Model config out of bounds", "errors": errors})


@api.get("/config/model", response_model=ModelConfig)
def get_model_config(user: Dict[str, Any] = Depends(get_current_user)):
    return _read_model_cfg()


@api.put("/config/model", response_model=ModelConfig)
def put_model_config(cfg: ModelConfig, user: Dict[str, Any] = Depends(get_current_user)):
    # Validate against system ranges
    ranges = _read_system_cfg()
    _validate_model_config(cfg, ranges)
    # Save to DB (single row upsert)
    try:
        payload = json.dumps(cfg.dict())
        db = SessionLocal()
        row = db.query(ModelConfigRow).filter(ModelConfigRow.id == 1).first()
        now = datetime.utcnow()
        if row:
            row.payload = payload
            row.updated_at = now
        else:
            row = ModelConfigRow(id=1, version=1, payload=payload, created_at=now, updated_at=now)
            db.add(row)
        db.commit()
        return cfg
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save model config: {e}")
    finally:
        try:
            db.close()
        except Exception:
            pass

# ------------------------------------------------------------------------------
# Minimal in-memory Mixed Games API to support the UI
# ------------------------------------------------------------------------------


from app.models.game import Game as DbGame, GameStatus as DbGameStatus
from app.models.user import UserTypeEnum


def _is_system_admin_user(user: Any) -> bool:
    if isinstance(user, dict):
        if user.get("is_superuser"):
            return True
        token = str(user.get("role") or user.get("user_type") or "").lower()
        return token in {"systemadmin", "system_admin", "superadmin", "systemadministrator"}

    if getattr(user, "is_superuser", False):
        return True

    user_type = getattr(user, "user_type", None)
    if isinstance(user_type, UserTypeEnum):
        return user_type == UserTypeEnum.SYSTEM_ADMIN
    if isinstance(user_type, str):
        token = user_type.lower()
        return token in {"systemadmin", "system_admin", "superadmin", "systemadministrator"}
    return False


def _extract_group_id(user: Any) -> Optional[int]:
    gid = user.get("group_id") if isinstance(user, dict) else getattr(user, "group_id", None)
    try:
        return int(gid) if gid is not None else None
    except (TypeError, ValueError):
        return None


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    if dt.tzinfo is None:
        return dt.isoformat() + "Z"
    return dt.isoformat()


STATUS_REMAPPING = {
    DbGameStatus.CREATED: "CREATED",
    DbGameStatus.STARTED: "IN_PROGRESS",
    DbGameStatus.ROUND_IN_PROGRESS: "IN_PROGRESS",
    DbGameStatus.ROUND_COMPLETED: "PAUSED",
    DbGameStatus.FINISHED: "COMPLETED",
}

PROGRESSION_SUPERVISED = "supervised"
PROGRESSION_UNSUPERVISED = "unsupervised"

ROLES_IN_ORDER = ["retailer", "wholesaler", "distributor", "manufacturer"]

ROLE_TO_AGENT_TYPE = {
    "retailer": AgentType.RETAILER,
    "wholesaler": AgentType.WHOLESALER,
    "distributor": AgentType.DISTRIBUTOR,
    "manufacturer": AgentType.MANUFACTURER,
}

_AUTO_ADVANCE_TASKS: Dict[int, asyncio.Task] = {}
_AUTO_TASKS_LOCK = threading.Lock()


def _role_key(player: Player) -> str:
    return str(player.role.value if hasattr(player.role, "value") else player.role).lower()


def _agent_type_for_role(role: str) -> AgentType:
    mapping = {
        "retailer": AgentType.RETAILER,
        "wholesaler": AgentType.WHOLESALER,
        "distributor": AgentType.DISTRIBUTOR,
        "manufacturer": AgentType.MANUFACTURER,
        "factory": AgentType.MANUFACTURER,
    }
    if role not in mapping:
        raise ValueError(f"Unsupported role '{role}'")
    return mapping[role]


def _strategy_for_player(player: Player) -> AgentStrategyEnum:
    raw = (player.ai_strategy or "daybreak_dtce").lower()
    try:
        return AgentStrategyEnum(raw)
    except ValueError:
        if raw.startswith("llm"):
            return AgentStrategyEnum.LLM
        return AgentStrategyEnum.DAYBREAK_DTCE


def _coerce_game_config(game: DbGame) -> Dict[str, Any]:
    raw = game.config or {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        token = value.strip().lower()
        return token in {"1", "true", "yes", "on", "enabled"}
    return False


def _normalize_debug_config(config: Dict[str, Any]) -> Dict[str, Any]:
    raw = config.get("debug_logging")
    if isinstance(raw, dict):
        cfg = dict(raw)
    elif isinstance(raw, bool):
        cfg = {"enabled": raw}
    else:
        cfg = {}
    enabled_token = cfg.get("enabled", cfg.get("active", cfg.get("debug")))
    cfg["enabled"] = _to_bool(enabled_token)
    return cfg


def _ensure_debug_log_file(config: Dict[str, Any], game: DbGame) -> Optional[Path]:
    cfg = _normalize_debug_config(config)
    if not cfg.get("enabled"):
        config["debug_logging"] = {"enabled": False}
        return None

    path_value = cfg.get("file_path")
    if path_value:
        path = Path(path_value)
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        name_token = (game.name or f"game_{game.id}")
        safe_token = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_" for char in name_token
        )[:48]
        filename = (
            f"game_{game.id}_{safe_token}_{timestamp}.txt"
            if safe_token
            else f"game_{game.id}_{timestamp}.txt"
        )
        path = DEBUG_LOG_DIR / filename
        cfg["file_path"] = str(path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("w", encoding="utf-8") as handle:
                handle.write(f"Game {game.id} Debug Log\n")
                if game.name:
                    handle.write(f"Name: {game.name}\n")
                handle.write(f"Created: {datetime.utcnow().isoformat()}Z\n\n")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to prepare debug log for game %s: %s", game.id, exc)
        cfg["enabled"] = False
        config["debug_logging"] = cfg
        return None

    config["debug_logging"] = cfg
    return path


def _format_debug_block(data: Any, *, indent: str = "      ") -> str:
    if data is None:
        return f"{indent}None"
    try:
        text = json.dumps(data, indent=2, sort_keys=True, default=str, ensure_ascii=False)
    except TypeError:
        text = json.dumps(str(data), ensure_ascii=False)
    return "\n".join(f"{indent}{line}" for line in text.splitlines())


def _append_debug_round_log(
    config: Dict[str, Any],
    game: DbGame,
    *,
    round_number: int,
    timestamp: datetime,
    entries: List[Dict[str, Any]],
) -> None:
    if not entries:
        return

    path = _ensure_debug_log_file(config, game)
    if not path:
        return

    iso_timestamp = timestamp.isoformat() + "Z"
    lines: List[str] = [f"Round {round_number} @ {iso_timestamp}"]
    for entry in entries:
        node_name = entry.get("node") or "unknown"
        lines.append(f"  Node: {node_name}")
        player_info = entry.get("player") or {}
        if player_info:
            player_label = player_info.get("name") or "Unnamed player"
            player_id = player_info.get("id")
            player_type = "AI" if player_info.get("is_ai") else "Human"
            if player_id is not None:
                lines.append(f"    Player: {player_label} (ID: {player_id}, {player_type})")
            else:
                lines.append(f"    Player: {player_label} ({player_type})")
        info_sent = entry.get("info_sent")
        lines.append("    Info provided:")
        lines.append(_format_debug_block(info_sent))
        reply = entry.get("reply")
        lines.append("    Reply:")
        lines.append(_format_debug_block(reply))
        ending_state = entry.get("ending_state")
        lines.append("    Ending state:")
        lines.append(_format_debug_block(ending_state))
    lines.append("")

    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to append debug log for game %s: %s", game.id, exc)


def _save_game_config(db: Session, game: DbGame, config: Dict[str, Any]) -> None:
    # Assign a shallow copy and flag the attribute as modified so SQLAlchemy
    # persists JSON changes even when only nested keys are updated.
    game.config = dict(config)
    flag_modified(game, "config")
    db.add(game)


def _get_progression_mode(game: DbGame) -> str:
    config = _coerce_game_config(game)
    mode = config.get("progression_mode", PROGRESSION_SUPERVISED)
    if mode not in {PROGRESSION_SUPERVISED, PROGRESSION_UNSUPERVISED}:
        return PROGRESSION_SUPERVISED
    return mode


def _ensure_round(db: Session, game: DbGame, round_number: Optional[int] = None) -> Round:
    number = round_number or (game.current_round or 1)
    existing = (
        db.query(Round)
        .filter(Round.game_id == game.id, Round.round_number == number)
        .first()
    )
    if existing:
        return existing
    round_record = Round(
        game_id=game.id,
        round_number=number,
        status="in_progress",
        started_at=datetime.utcnow(),
        config={},
    )
    db.add(round_record)
    db.flush()
    return round_record


def _pending_orders(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.setdefault("pending_orders", {})


def _simulation_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("simulation_parameters", {})


def _all_players_submitted(db: Session, game: DbGame, round_record: Round) -> bool:
    player_roles = {
        str(player.role.value if hasattr(player.role, "value") else player.role).lower()
        for player in db.query(Player).filter(Player.game_id == game.id).all()
    }
    if not player_roles:
        return False

    actions = (
        db.query(PlayerAction, Player)
        .join(Player, Player.id == PlayerAction.player_id)
        .filter(
            PlayerAction.game_id == game.id,
            PlayerAction.round_id == round_record.id,
            PlayerAction.action_type == "order",
        )
        .all()
    )
    submitted_roles = {
        str(player.role.value if hasattr(player.role, "value") else player.role).lower()
        for _, player in actions
        if _.quantity is not None
    }
    return player_roles.issubset(submitted_roles)


def _compute_customer_demand(game: DbGame, round_number: int) -> int:
    config = _coerce_game_config(game)
    params = _simulation_parameters(config)
    initial = int(params.get("initial_demand", 4))
    change_week = int(params.get("demand_change_week", params.get("change_week", 20)))
    new_demand = int(params.get("new_demand", params.get("final_demand", initial)))

    pattern = config.get("demand_pattern", {})
    pattern_type = str(pattern.get("type", "classic")).lower()
    pattern_params = pattern.get("params", {})

    if pattern_type == "constant":
        return int(pattern_params.get("value", initial))
    if pattern_type == "classic":
        initial = int(pattern_params.get("initial_demand", initial))
        change_week = int(pattern_params.get("change_week", change_week))
        new_demand = int(pattern_params.get("final_demand", new_demand))
        return initial if round_number < change_week else new_demand

    if pattern_type == "seasonal":
        period = max(1, int(pattern_params.get("period", 4)))
        amplitude = float(pattern_params.get("amplitude", 2))
        base = float(pattern_params.get("base", initial))
        import math

        return max(0, int(base + amplitude * math.sin(2 * math.pi * round_number / period)))

    return initial if round_number < change_week else new_demand


def _ensure_simulation_state(config: Dict[str, Any]) -> Dict[str, Any]:
    node_policies = config.get("node_policies", {})
    lane_views = MixedGameService._build_lane_views(node_policies, config)
    all_nodes = lane_views.get("all_nodes") or [MixedGameService._normalise_key(name) for name in node_policies.keys()]

    engine = config.setdefault("engine_state", {})
    for node in all_nodes:
        MixedGameService._ensure_engine_node(engine, node_policies, node)

    state = config.setdefault("simulation_state", {})
    state.setdefault("inventory", {})
    state.setdefault("backlog", {})
    state.setdefault("last_orders", {})
    state.setdefault("incoming_shipments", {})
    state.setdefault("pending_orders", {})

    # Remove legacy pipeline keys that are no longer used
    state.pop("ship_pipeline", None)
    state.pop("order_pipeline", None)
    state.pop("production_pipeline", None)

    if "initial_state" not in config:
        config["initial_state"] = {
            role: {
                "inventory": int(engine.get(role, {}).get("inventory", 0)),
                "backlog": int(engine.get(role, {}).get("backlog", 0)),
            }
            for role in ROLES_IN_ORDER
        }

    return state


def _finalize_round_if_ready(
    db: Session,
    game: DbGame,
    config: Dict[str, Any],
    round_record: Round,
    *,
    force: bool = False,
) -> bool:
    pending = _pending_orders(config)
    if not force and _get_progression_mode(game) != PROGRESSION_UNSUPERVISED:
        return False

    if not force and not _all_players_submitted(db, game, round_record):
        return False

    timestamp = datetime.utcnow()
    round_number = round_record.round_number
    external_demand = _compute_customer_demand(game, round_number)

    players = db.query(Player).filter(Player.game_id == game.id).all()
    players_by_role = {
        str(player.role.value if hasattr(player.role, "value") else player.role).lower(): player
        for player in players
    }

    agent_manager = AgentManager()
    configured_agents: Set[str] = set()
    overrides = config.get("daybreak_overrides") or {}
    info_sharing_cfg = config.get("info_sharing") or {}
    full_visibility = str(info_sharing_cfg.get("visibility", "")).lower() == "full"

    actions = (
        db.query(PlayerAction, Player)
        .join(Player, Player.id == PlayerAction.player_id)
        .filter(
            PlayerAction.game_id == game.id,
            PlayerAction.round_id == round_record.id,
            PlayerAction.action_type == "order",
        )
        .all()
    )

    orders_by_role = {}
    actions_by_role = {}
    for action, player in actions:
        role_key = str(player.role.value if hasattr(player.role, "value") else player.role).lower()
        pending_entry = pending.get(role_key, {}) if pending else {}
        orders_by_role[role_key] = {
            "player_id": player.id,
            "quantity": int(action.quantity or 0),
            "comment": pending_entry.get("comment"),
            "submitted_at": pending_entry.get("submitted_at") or action.created_at.isoformat() + "Z",
        }
        actions_by_role[role_key] = action

    state = _ensure_simulation_state(config)
    node_policies = config.get("node_policies", {})
    global_policy = config.get("global_policy", {})
    lane_views = MixedGameService._build_lane_views(node_policies, config)
    node_types_map = lane_views.get("node_types", {})
    all_nodes = lane_views.get("all_nodes") or [MixedGameService._normalise_key(name) for name in node_policies.keys()]

    engine = config.setdefault("engine_state", {})
    for node in all_nodes:
        MixedGameService._ensure_engine_node(engine, node_policies, node)

    market_demand_nodes_config = {n for n in lane_views.get("market_nodes", []) if n in engine}
    market_demand_nodes_types = {node for node, node_type in node_types_map.items() if node_type == "market_demand"}
    market_demand_nodes = market_demand_nodes_config or market_demand_nodes_types
    if not market_demand_nodes and all_nodes:
        market_demand_nodes = {all_nodes[-1]}

    pre_inventory = {node: engine[node].get("inventory", 0) for node in all_nodes}
    pre_backlog = {node: engine[node].get("backlog", 0) for node in all_nodes}
    pre_costs = {
        node: {
            "holding_cost": engine[node].get("holding_cost", 0.0),
            "backorder_cost": engine[node].get("backorder_cost", 0.0),
        }
        for node in all_nodes
    }
    previous_orders_by_node = {node: engine[node].get("last_order", 0) for node in all_nodes}

    pending_demand = defaultdict(int)
    for node in market_demand_nodes:
        if node in engine:
            pending_demand[node] += int(external_demand)

    for lane in lane_views.get("lanes", []):
        upstream = lane["from"]
        downstream = lane["to"]
        if upstream not in engine or downstream not in engine:
            continue
        downstream_type = node_types_map.get(downstream, "")
        downstream_state = engine.get(downstream, {})
        if downstream_type == "market_demand":
            downstream_order = pending_demand.get(downstream, downstream_state.get("last_order", 0))
        else:
            downstream_order = downstream_state.get("last_order", 0)
        try:
            downstream_value = int(downstream_order or 0)
        except (TypeError, ValueError):
            downstream_value = 0
        pending_demand[upstream] += downstream_value

    incoming_orders_map = {}
    for node in all_nodes:
        state_node = engine[node]
        policy = MixedGameService._policy_for_node(node_policies, node)
        info_delay = max(0, int(policy.get("info_delay", 0)))
        new_demand = int(pending_demand.get(node, 0))
        if info_delay <= 0:
            incoming = new_demand
        else:
            queue = state_node.setdefault("info_queue", [0] * info_delay)
            queue.append(new_demand)
            incoming = queue.pop(0) if queue else 0
        state_node["incoming_orders"] = incoming
        incoming_orders_map[node] = incoming

    hold_cost = float(global_policy.get("holding_cost", 0.5))
    back_cost = float(global_policy.get("backlog_cost", 1.0))

    demand_totals = {node: engine[node]["backlog"] + incoming_orders_map.get(node, 0) for node in all_nodes}
    inventory_available = {node: engine[node]["inventory"] for node in all_nodes}

    shipments_inbound = defaultdict(int)
    shipments_planned = defaultdict(int)
    node_sequence = lane_views.get("node_sequence") or all_nodes
    lanes_by_upstream = lane_views.get("lanes_by_upstream", {})

    for upstream in node_sequence:
        upstream_type = node_types_map.get(upstream, "")
        for lane in lanes_by_upstream.get(upstream, []):
            downstream = lane["to"]
            if downstream not in demand_totals:
                continue
            remaining_demand = max(0, demand_totals[downstream] - shipments_inbound[downstream])
            if remaining_demand <= 0:
                continue
            capacity = lane.get("capacity")
            if capacity is not None:
                try:
                    remaining_demand = min(remaining_demand, int(capacity))
                except (TypeError, ValueError):
                    pass
            if upstream_type == "market_supply":
                ship_qty = remaining_demand
            else:
                available = inventory_available.get(upstream, 0)
                if available <= 0:
                    continue
                ship_qty = min(available, remaining_demand)
            if ship_qty <= 0:
                continue
            if upstream_type != "market_supply":
                inventory_available[upstream] -= ship_qty
            shipments_inbound[downstream] += ship_qty
            shipments_planned[upstream] += ship_qty

    for node, remaining in inventory_available.items():
        engine[node]["inventory"] = remaining

    arrivals_map = {}
    for node in all_nodes:
        state_node = engine[node]
        policy = MixedGameService._policy_for_node(node_policies, node)
        ship_delay = max(0, int(policy.get("ship_delay", 0)))
        planned = int(shipments_inbound.get(node, 0))
        if ship_delay <= 0:
            arriving = planned
        else:
            queue = state_node.setdefault("ship_queue", [0] * ship_delay)
            queue.append(planned)
            arriving = queue.pop(0) if queue else 0
        state_node["incoming_shipments"] = arriving
        state_node["last_shipment_planned"] = int(shipments_planned.get(node, 0))
        arrivals_map[node] = arriving

    for node in all_nodes:
        node_type = node_types_map.get(node, "")
        state_node = engine[node]
        arriving = arrivals_map.get(node, 0)
        if node_type == "market_supply":
            state_node["inventory"] = 0
            state_node["backlog"] = 0
            state_node["last_arrival"] = arriving
            state_node["holding_cost"] = 0.0
            state_node["backorder_cost"] = 0.0
            state_node["total_cost"] = 0.0
            continue
        available_inventory = max(0, state_node.get("inventory", 0)) + arriving
        demand_here = demand_totals.get(node, 0)
        shipped = min(available_inventory, demand_here)
        state_node["inventory"] = available_inventory - shipped
        state_node["backlog"] = max(0, demand_here - shipped)
        state_node["last_arrival"] = arriving
        if node_type == "market_demand":
            state_node["holding_cost"] = 0.0
            state_node["backorder_cost"] = state_node["backlog"] * back_cost
            state_node["total_cost"] = state_node["backorder_cost"]
        else:
            state_node["holding_cost"] = state_node.get("holding_cost", 0.0) + state_node["inventory"] * hold_cost
            state_node["backorder_cost"] = state_node.get("backorder_cost", 0.0) + state_node["backlog"] * back_cost
            state_node["total_cost"] = state_node["holding_cost"] + state_node["backorder_cost"]

    shipments_map = lane_views.get("shipments_map", {})
    node_orders_new = {}
    orders_timestamp_iso = timestamp.isoformat() + "Z"

    round_debug_entries: Dict[str, Dict[str, Any]] = {}
    round_debug_order: List[str] = []

    processing_nodes = []
    for node in reversed(node_sequence):
        if node not in processing_nodes:
            processing_nodes.append(node)
    for node in all_nodes:
        if node not in processing_nodes:
            processing_nodes.append(node)

    for node_key in processing_nodes:
        player = players_by_role.get(node_key)
        node_type = node_types_map.get(node_key, "")
        if not player or node_type in {"market_demand", "market_supply"}:
            continue
        agent_type = ROLE_TO_AGENT_TYPE.get(node_key)
        if not agent_type:
            continue
        if node_key not in configured_agents:
            strategy_value = (player.ai_strategy or "naive").lower()
            try:
                strategy_enum = AgentStrategyEnum(strategy_value)
            except ValueError:
                if strategy_value.startswith("llm"):
                    strategy_enum = AgentStrategyEnum.LLM
                else:
                    strategy_enum = AgentStrategyEnum.NAIVE
            override_pct = overrides.get(node_key)
            agent_manager.set_agent_strategy(
                agent_type,
                strategy_enum,
                llm_model=player.llm_model,
                override_pct=override_pct,
            )
            configured_agents.add(node_key)
        agent = agent_manager.get_agent(agent_type)
        downstream_nodes = shipments_map.get(node_key, [])
        downstream_orders_latest = {
            MixedGameService._normalise_key(down): node_orders_new.get(down, 0)
            for down in downstream_nodes
        }
        previous_downstream_orders = {
            MixedGameService._normalise_key(down): previous_orders_by_node.get(down, 0)
            for down in downstream_nodes
        }
        previous_orders_seq = list(previous_downstream_orders.values())
        node_state = engine[node_key]
        local_state = {
            "inventory": int(node_state.get("inventory", 0)),
            "backlog": int(node_state.get("backlog", 0)),
            "incoming_shipments": int(arrivals_map.get(node_key, 0)),
        }
        if full_visibility:
            local_state["downstream_orders"] = list(downstream_orders_latest.values())
        observed_demand = int(incoming_orders_map.get(node_key, 0))
        if node_key in market_demand_nodes:
            current_demand_value = int(external_demand)
        elif player.can_see_demand or full_visibility:
            current_demand_value = observed_demand
        else:
            current_demand_value = None
        upstream_context = {
            "previous_orders": previous_orders_seq,
            "previous_orders_by_role": previous_downstream_orders,
            "downstream_orders": downstream_orders_latest,
        }
        info_payload = {
            "current_round": round_number,
            "current_demand": current_demand_value,
            "local_state": dict(local_state),
            "upstream_context": upstream_context,
        }
        player_info = {
            "id": getattr(player, "id", None),
            "name": getattr(player, "name", None),
            "is_ai": bool(getattr(player, "is_ai", False)),
        }
        order_qty = agent.make_decision(
            current_round=round_number,
            current_demand=current_demand_value,
            upstream_data=upstream_context,
            local_state=local_state,
        )
        order_qty = max(0, int(round(order_qty)))
        node_orders_new[node_key] = order_qty
        node_state["last_order"] = order_qty
        node_state["on_order"] = max(
            0,
            node_state.get("on_order", 0) + order_qty - node_state.get("incoming_shipments", 0),
        )
        entry = orders_by_role.get(node_key, {
            "player_id": player.id,
            "quantity": order_qty,
            "comment": None,
            "submitted_at": orders_timestamp_iso,
        })
        entry["player_id"] = player.id
        entry["quantity"] = order_qty
        entry.setdefault("submitted_at", orders_timestamp_iso)
        orders_by_role[node_key] = entry
        reply_payload = {
            "type": "agent_decision",
            "order_quantity": order_qty,
            "submitted_at": entry.get("submitted_at"),
        }
        if entry.get("comment"):
            reply_payload["comment"] = entry["comment"]
        debug_entry = {
            "node": node_key,
            "player": player_info,
            "info_sent": info_payload,
            "reply": reply_payload,
        }
        round_debug_entries[node_key] = debug_entry
        if node_key not in round_debug_order:
            round_debug_order.append(node_key)
        action_obj = actions_by_role.get(node_key)
        if action_obj is None:
            action_obj = PlayerAction(
                game_id=game.id,
                round_id=round_record.id,
                player_id=player.id,
                action_type="order",
                quantity=order_qty,
                created_at=timestamp,
            )
            db.add(action_obj)
        else:
            action_obj.quantity = order_qty
            action_obj.created_at = timestamp
        actions_by_role[node_key] = action_obj

    for role in ROLES_IN_ORDER:
        orders_by_role.setdefault(
            role,
            {
                "player_id": players_by_role.get(role).id if players_by_role.get(role) else None,
                "quantity": engine.get(role, {}).get("last_order", 0),
                "comment": None,
                "submitted_at": orders_timestamp_iso,
            },
        )

    for node in all_nodes:
        if node in round_debug_entries:
            continue
        player = players_by_role.get(node)
        if not player:
            continue
        recorded = orders_by_role.get(node, {})
        reply_payload = {
            "type": "human_input"
            if not bool(getattr(player, "is_ai", False))
            else "recorded_order",
            "order_quantity": int(recorded.get("quantity") or 0),
            "submitted_at": recorded.get("submitted_at"),
        }
        if recorded.get("comment"):
            reply_payload["comment"] = recorded["comment"]
        info_payload = {
            "current_round": round_number,
            "note": "Order received from player submission",
            "observed_demand": int(incoming_orders_map.get(node, 0)),
            "inventory_before": int(
                pre_inventory.get(node, engine.get(node, {}).get("inventory", 0))
            ),
            "backlog_before": int(
                pre_backlog.get(node, engine.get(node, {}).get("backlog", 0))
            ),
        }
        debug_entry = {
            "node": node,
            "player": {
                "id": getattr(player, "id", None),
                "name": getattr(player, "name", None),
                "is_ai": bool(getattr(player, "is_ai", False)),
            },
            "info_sent": info_payload,
            "reply": reply_payload,
        }
        round_debug_entries[node] = debug_entry
        if node not in round_debug_order:
            round_debug_order.append(node)

    role_stats = {}
    for role in ROLES_IN_ORDER:
        node_state = engine.get(role, {})
        before_inv = pre_inventory.get(role, 0)
        before_backlog = pre_backlog.get(role, 0)
        inv_after = node_state.get("inventory", 0)
        backlog_after = node_state.get("backlog", 0)
        order_qty = node_state.get("last_order", 0)
        shipped_qty = shipments_planned.get(role, 0)
        arrivals_qty = arrivals_map.get(role, 0)
        demand_here = incoming_orders_map.get(role, 0)
        holding_cost_delta = float(node_state.get("holding_cost", 0.0) - pre_costs.get(role, {}).get("holding_cost", 0.0))
        backlog_cost_delta = float(node_state.get("backorder_cost", 0.0) - pre_costs.get(role, {}).get("backorder_cost", 0.0))
        total_cost_delta = holding_cost_delta + backlog_cost_delta
        role_stats[role] = {
            "inventory_before": before_inv,
            "inventory_after": inv_after,
            "backlog_before": before_backlog,
            "backlog_after": backlog_after,
            "demand": demand_here,
            "shipped": shipped_qty,
            "arrivals": arrivals_qty,
            "order": order_qty,
            "holding_cost": round(holding_cost_delta, 2),
            "backlog_cost": round(backlog_cost_delta, 2),
            "total_cost": round(total_cost_delta, 2),
        }

    for node, debug_entry in round_debug_entries.items():
        stats = role_stats.get(node)
        node_state = engine.get(node, {})
        ending_state = {
            "inventory": int(
                node_state.get("inventory", stats["inventory_after"] if stats else 0)
            ),
            "backlog": int(
                node_state.get("backlog", stats["backlog_after"] if stats else 0)
            ),
            "last_order": int(node_state.get("last_order", 0)),
            "on_order": int(node_state.get("on_order", 0)),
            "incoming_shipments": int(node_state.get("incoming_shipments", 0)),
            "observed_demand": int(
                incoming_orders_map.get(node, stats["demand"] if stats else 0)
            ),
        }
        if stats:
            ending_state.update(
                {
                    "shipped": int(stats.get("shipped", 0)),
                    "arrivals": int(stats.get("arrivals", 0)),
                    "holding_cost": stats.get("holding_cost"),
                    "backlog_cost": stats.get("backlog_cost"),
                    "total_cost_delta": stats.get("total_cost"),
                }
            )
        debug_entry["ending_state"] = ending_state

    state_inventory = state.setdefault("inventory", {})
    state_backlog = state.setdefault("backlog", {})
    state_last_orders = state.setdefault("last_orders", {})
    state_incoming = state.setdefault("incoming_shipments", {})
    state_pending_orders = state.setdefault("pending_orders", {})
    for role in ROLES_IN_ORDER:
        state_inventory[role] = int(role_stats[role]["inventory_after"])
        state_backlog[role] = int(role_stats[role]["backlog_after"])
        state_last_orders[role] = int(role_stats[role]["order"])
        state_incoming[role] = int(role_stats[role]["arrivals"])
        state_pending_orders[role] = int(role_stats[role]["demand"])

    if pending is not None:
        pending.clear()

    ordered_debug_entries: List[Dict[str, Any]] = []
    for node in round_debug_order:
        entry = round_debug_entries.get(node)
        if entry:
            ordered_debug_entries.append(entry)
    for node, entry in round_debug_entries.items():
        if entry not in ordered_debug_entries:
            ordered_debug_entries.append(entry)

    history_entry = {
        "round": round_number,
        "timestamp": timestamp.isoformat() + "Z",
        "demand": external_demand,
        "orders": {role: dict(orders_by_role.get(role, {})) for role in ROLES_IN_ORDER},
        "inventory_positions": {role: role_stats[role]["inventory_after"] for role in ROLES_IN_ORDER},
        "backlogs": {role: role_stats[role]["backlog_after"] for role in ROLES_IN_ORDER},
        "node_states": {
            role: {
                "inventory_before": role_stats[role]["inventory_before"],
                "inventory_after": role_stats[role]["inventory_after"],
                "backlog_before": role_stats[role]["backlog_before"],
                "backlog_after": role_stats[role]["backlog_after"],
                "incoming_order": role_stats[role]["demand"],
                "arrivals": role_stats[role]["arrivals"],
                "shipped": role_stats[role]["shipped"],
                "last_order": role_stats[role]["order"],
            }
            for role in ROLES_IN_ORDER
        },
        "costs": {
            role: {
                "holding_cost": role_stats[role]["holding_cost"],
                "backlog_cost": role_stats[role]["backlog_cost"],
                "total_cost": role_stats[role]["total_cost"],
            }
            for role in ROLES_IN_ORDER
        },
        "total_cost": round(sum(role_stats[role]["total_cost"] for role in ROLES_IN_ORDER), 2),
    }

    _append_debug_round_log(
        config,
        game,
        round_number=round_number,
        timestamp=timestamp,
        entries=ordered_debug_entries,
    )

    config.setdefault("history", []).append(history_entry)
    config["pending_orders"] = {}

    round_record.status = "completed"
    round_record.completed_at = timestamp
    round_record.config = {
        "orders": {role: dict(orders_by_role.get(role, {})) for role in ROLES_IN_ORDER},
        "demand": external_demand,
        "node_states": history_entry.get("node_states", {}),
    }

    if game.max_rounds and round_number >= game.max_rounds:
        game.status = DbGameStatus.FINISHED
        game.current_round = round_number
    else:
        game.current_round = round_number + 1
        next_round = _ensure_round(db, game, game.current_round)
        next_round.status = "in_progress"
        next_round.started_at = datetime.utcnow()
        game.status = (
            DbGameStatus.ROUND_IN_PROGRESS
            if _get_progression_mode(game) == PROGRESSION_UNSUPERVISED
            else DbGameStatus.STARTED
        )

    _touch_game(game)
    _save_game_config(db, game, config)
    db.add(round_record)
    db.add(game)
    db.flush()
    return True


def _auto_advance_unsupervised_game_sync(
    game_id: int,
    *,
    sleep_seconds: float = 0.35,
    iteration_limit: int = 2048,
) -> None:
    session = SyncSessionLocal()
    try:
        game = session.query(DbGame).filter(DbGame.id == game_id).first()
        if not game:
            logger.warning("Auto-advance aborted: game %s not found", game_id)
            return

        if _get_progression_mode(game) != PROGRESSION_UNSUPERVISED:
            logger.debug("Game %s is not unsupervised; skipping auto advance", game_id)
            return

        players = session.query(Player).filter(Player.game_id == game.id).all()
        if not players:
            logger.warning("Auto-advance aborted: no players for game %s", game_id)
            return

        if any(not p.is_ai for p in players):
            logger.info(
                "Auto-advance skipped for game %s because at least one player is human",
                game_id,
            )
            return

        iteration = 0

        while True:
            iteration += 1
            if iteration_limit and iteration > iteration_limit:
                logger.warning(
                    "Auto-advance stopped for game %s after reaching the iteration limit",
                    game_id,
                )
                break

            session.expire_all()
            game = session.query(DbGame).filter(DbGame.id == game_id).first()
            if not game:
                logger.debug("Game %s disappeared mid-run; stopping auto advance", game_id)
                break

            if _get_progression_mode(game) != PROGRESSION_UNSUPERVISED:
                logger.debug(
                    "Game %s progression mode changed away from unsupervised; stopping",
                    game_id,
                )
                break

            if game.status == DbGameStatus.FINISHED:
                break

            if game.current_round is None or game.current_round <= 0:
                game.current_round = 1

            round_record = _ensure_round(session, game, game.current_round)
            if round_record.status != "in_progress":
                round_record.status = "in_progress"
                round_record.started_at = datetime.utcnow()

            config = _coerce_game_config(game)
            _ensure_simulation_state(config)
            pending = _pending_orders(config)
            pending.clear()

            session.flush()
            _save_game_config(session, game, config)
            progressed = _finalize_round_if_ready(session, game, config, round_record, force=True)
            session.add(game)
            session.commit()

            if not progressed:
                logger.debug(
                    "Auto-advance fallback triggered for game %s round %s",
                    game_id,
                    round_record.round_number,
                )
                game.current_round = (game.current_round or 0) + 1
                session.commit()

            if game.status == DbGameStatus.FINISHED:
                break

            if sleep_seconds:
                time.sleep(sleep_seconds)
    except Exception:
        logger.exception("Auto-advance failed for game %s", game_id)
    finally:
        session.close()


async def _schedule_unsupervised_autoplay(game_id: int) -> None:
    with _AUTO_TASKS_LOCK:
        existing = _AUTO_ADVANCE_TASKS.get(game_id)
        if existing and not existing.done():
            return

    async def runner() -> None:
        current_task = asyncio.current_task()
        try:
            await asyncio.to_thread(_auto_advance_unsupervised_game_sync, game_id)
        finally:
            with _AUTO_TASKS_LOCK:
                if _AUTO_ADVANCE_TASKS.get(game_id) is current_task:
                    _AUTO_ADVANCE_TASKS.pop(game_id, None)

    task = asyncio.create_task(runner())
    with _AUTO_TASKS_LOCK:
        _AUTO_ADVANCE_TASKS[game_id] = task
    logger.info("Scheduled unsupervised auto-advance for game %s", game_id)


def _cancel_unsupervised_autoplay(game_id: int) -> None:
    with _AUTO_TASKS_LOCK:
        task = _AUTO_ADVANCE_TASKS.pop(game_id, None)
    if task and not task.done():
        task.cancel()
        # Cancellation is cooperative; once cancelled, the background task will
        # exit on its own when the event loop next runs. No further action needed.


def _replay_history_from_rounds(db: Session, game: DbGame) -> List[Dict[str, Any]]:
    records = (
        db.query(Round)
        .filter(Round.game_id == game.id)
        .order_by(Round.round_number.asc())
        .all()
    )

    history: List[Dict[str, Any]] = []
    for record in records:
        payload: Dict[str, Any] = {}
        if record.config:
            if isinstance(record.config, dict):
                payload = dict(record.config)
            else:
                try:
                    payload = json.loads(record.config)
                except json.JSONDecodeError:
                    payload = {}

        history.append(
            {
                "round": record.round_number,
                "timestamp": _iso(record.completed_at) if record.completed_at else None,
                "demand": payload.get("demand"),
                "orders": payload.get("orders", {}),
                "node_states": payload.get("node_states", {}),
            }
        )

    return history


def _compute_game_report(db: Session, game: DbGame) -> Dict[str, Any]:
    config = _coerce_game_config(game)
    history = list(config.get("history", []))
    if not history:
        history = _replay_history_from_rounds(db, game)

    totals: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    order_series: Dict[str, List[Dict[str, Any]]] = {}
    inventory_series: List[Dict[str, Any]] = []
    demand_series: List[Dict[str, Any]] = []

    for entry in history:
        round_number = entry.get("round")
        demand_series.append({"round": round_number, "demand": entry.get("demand", 0)})

        inv_snapshot = {"round": round_number}
        for role in ROLES_IN_ORDER:
            order_info = entry.get("orders", {}).get(role, {})
            qty = order_info.get("quantity", 0)
            order_series.setdefault(role, []).append({"round": round_number, "quantity": qty})

            cost_info = entry.get("costs", {}).get(role, {})
            totals_role = totals[role]
            totals_role["holding_cost"] += float(cost_info.get("holding_cost", 0))
            totals_role["backlog_cost"] += float(cost_info.get("backlog_cost", 0))
            totals_role["total_cost"] += float(cost_info.get("total_cost", 0))
            totals_role["orders"] += float(qty)

            inv_value = entry.get("inventory_positions", {}).get(role, 0)
            backlog_value = entry.get("backlogs", {}).get(role, 0)
            inv_snapshot[role] = inv_value
            totals_role["final_inventory"] = inv_value
            totals_role["final_backlog"] = backlog_value

        inventory_series.append(inv_snapshot)

    total_cost = sum(role_totals["total_cost"] for role_totals in totals.values())
    formatted_totals = {
        role: {
            "holding_cost": round(values.get("holding_cost", 0), 2),
            "backlog_cost": round(values.get("backlog_cost", 0), 2),
            "total_cost": round(values.get("total_cost", 0), 2),
            "orders": round(values.get("orders", 0), 2),
            "final_inventory": values.get("final_inventory", 0),
            "final_backlog": values.get("final_backlog", 0),
        }
        for role, values in totals.items()
    }

    # Ensure we always expose all roles, even if no rounds were recorded
    if not history:
        for role in ROLES_IN_ORDER:
            totals.setdefault(role, defaultdict(float))

    return {
        "game_id": game.id,
        "name": game.name,
        "progression_mode": _get_progression_mode(game),
        "total_cost": round(total_cost, 2),
        "totals": formatted_totals,
        "history": history,
        "order_series": order_series,
        "inventory_series": inventory_series,
        "demand_series": demand_series,
        "rounds_completed": len(history),
    }


def _serialize_game(game: DbGame) -> Dict[str, Any]:
    if isinstance(game.status, DbGameStatus):
        status = STATUS_REMAPPING.get(game.status, game.status.value)
    else:
        status = str(game.status or "")
    config = _coerce_game_config(game)
    demand_pattern = game.demand_pattern or config.get("demand_pattern") or {}
    return {
        "id": game.id,
        "name": game.name,
        "description": game.description,
        "status": str(status).upper(),
        "current_round": game.current_round or 0,
        "max_rounds": game.max_rounds or 0,
        "created_at": _iso(game.created_at),
        "updated_at": _iso(getattr(game, "updated_at", None)),
        "group_id": game.group_id,
        "created_by": game.created_by,
        "is_public": bool(getattr(game, "is_public", True)),
        "config": config,
        "demand_pattern": demand_pattern,
        "progression_mode": _get_progression_mode(game),
    }


def _get_game_for_user(db: Session, user: Any, game_id: int) -> DbGame:
    game = db.query(DbGame).filter(DbGame.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    if _is_system_admin_user(user):
        return game
    user_group_id = _extract_group_id(user)
    if user_group_id is None or user_group_id != game.group_id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return game


def _touch_game(game: DbGame) -> None:
    game.updated_at = datetime.utcnow()


@api.post("/mixed-games/", status_code=201)
async def create_mixed_game(payload: GameCreate, user: Dict[str, Any] = Depends(get_current_user)):
    db = SyncSessionLocal()
    try:
        group_id = _extract_group_id(user)
        if group_id is None and not _is_system_admin_user(user):
            raise HTTPException(status_code=403, detail="Group membership required to create games")
        if payload.supply_chain_config_id is None:
            raise HTTPException(status_code=400, detail="supply_chain_config_id is required to create a mixed game")

        config: Dict[str, Any] = {
            "demand_pattern": payload.demand_pattern.dict() if payload.demand_pattern else {},
            "pricing_config": payload.pricing_config.dict() if payload.pricing_config else {},
            "node_policies": {key: policy.dict() for key, policy in (payload.node_policies or {}).items()},
            "system_config": payload.system_config or {},
            "global_policy": payload.global_policy or {},
            "progression_mode": payload.progression_mode,
            "pending_orders": {},
            "history": [],
        }

        game = DbGame(
            name=payload.name,
            description=payload.description,
            status=DbGameStatus.CREATED,
            current_round=0,
            max_rounds=payload.max_rounds or 0,
            created_at=datetime.utcnow(),
            group_id=group_id,
            created_by=user.get("id"),
            is_public=payload.is_public,
            demand_pattern=config.get("demand_pattern", {}),
            config=config,
            supply_chain_config_id=int(payload.supply_chain_config_id),
        )
        db.add(game)
        db.flush()

        role_enum_map = {
            "retailer": "RETAILER",
            "wholesaler": "WHOLESALER",
            "distributor": "DISTRIBUTOR",
            "manufacturer": "MANUFACTURER",
            "factory": "MANUFACTURER",
        }

        for assignment in payload.player_assignments:
            is_agent = assignment.player_type == PlayerTypeSchema.AGENT
            role_value_raw = assignment.role.value if hasattr(assignment.role, "value") else str(assignment.role)
            role_key = role_value_raw.lower()
            role_enum_name = role_enum_map.get(role_key)
            if not role_enum_name:
                raise HTTPException(status_code=400, detail=f"Unsupported role: {role_value_raw}")
            if not is_agent and assignment.user_id is None:
                raise HTTPException(status_code=400, detail=f"User ID required for human role {role_value_raw}")

            strategy_value = None
            if assignment.strategy is not None:
                strategy_value = assignment.strategy.value if hasattr(assignment.strategy, "value") else str(assignment.strategy)

            player = Player(
                game_id=game.id,
                user_id=None if is_agent else assignment.user_id,
                name=f"{role_enum_name.title().replace('_', ' ')} ({'AI' if is_agent else 'Human'})",
                role=PlayerRole[role_enum_name],
                is_ai=is_agent,
                ai_strategy=strategy_value,
                can_see_demand=assignment.can_see_demand,
                llm_model=assignment.llm_model if is_agent else None,
                strategy=PlayerStrategy.MANUAL,
            )
            db.add(player)

        _ensure_simulation_state(config)
        _save_game_config(db, game, config)
        db.add(game)
        db.commit()
        db.refresh(game)
        payload = _serialize_game(game)
        if _get_progression_mode(game) == PROGRESSION_UNSUPERVISED:
            await _schedule_unsupervised_autoplay(game.id)
        return payload
    finally:
        db.close()


def _apply_player_assignments_for_update(
    db: Session,
    game: DbGame,
    assignments: List[PlayerAssignment],
    config: Dict[str, Any],
) -> None:
    existing_players = {
        (player.role.name if hasattr(player.role, "name") else str(player.role).upper()): player
        for player in db.query(Player).filter(Player.game_id == game.id).all()
    }

    role_assignments: Dict[str, Dict[str, Any]] = {}
    overrides: Dict[str, float] = {}

    for assignment in assignments:
        role_value = assignment.role.value if hasattr(assignment.role, "value") else str(assignment.role)
        role_key = role_value.lower()
        role_enum_name = role_value.upper()
        if role_enum_name not in PlayerRole.__members__:
            raise HTTPException(status_code=400, detail=f"Unsupported role: {role_value}")

        role_enum = PlayerRole[role_enum_name]
        player = existing_players.pop(role_enum.name, None)
        if player is None:
            player = Player(
                game_id=game.id,
                role=role_enum,
                name=role_enum.name.title(),
            )

        is_agent = assignment.player_type == PlayerTypeSchema.AGENT
        strategy_value = None
        if assignment.strategy is not None:
            strategy_value = (
                assignment.strategy.value
                if hasattr(assignment.strategy, "value")
                else str(assignment.strategy)
            )

        player.is_ai = is_agent
        player.type = PlayerModelType.AI if is_agent else PlayerModelType.HUMAN
        player.ai_strategy = strategy_value if is_agent else None
        player.can_see_demand = assignment.can_see_demand
        player.user_id = None if is_agent else assignment.user_id
        player.llm_model = assignment.llm_model if is_agent and assignment.llm_model else None
        player.name = f"{role_enum.name.title()} ({'AI' if is_agent else 'Human'})"
        player.strategy = PlayerStrategy.MANUAL
        db.add(player)

        role_assignments[role_key] = {
            "is_ai": is_agent,
            "agent_config_id": None,
            "user_id": None if is_agent else assignment.user_id,
        }

        if assignment.daybreak_override_pct is not None:
            overrides[role_key] = float(assignment.daybreak_override_pct)

    for leftover in existing_players.values():
        db.delete(leftover)

    if overrides:
        config["daybreak_overrides"] = overrides
    else:
        config.pop("daybreak_overrides", None)

    game.role_assignments = role_assignments

@api.get("/mixed-games/")
async def list_mixed_games(user: Dict[str, Any] = Depends(get_current_user)):
    db = SyncSessionLocal()
    try:
        query = db.query(DbGame).order_by(DbGame.created_at.desc())
        if not _is_system_admin_user(user):
            group_id = _extract_group_id(user)
            if group_id is None:
                return []
            query = query.filter(DbGame.group_id == group_id)
        games = query.all()
        if not games:
            ensure_default_group_and_game(db)
            db.expire_all()
            games = query.all()

        payload: List[Dict[str, Any]] = []
        for game in games:
            if _get_progression_mode(game) == PROGRESSION_UNSUPERVISED and game.status in {
                DbGameStatus.STARTED,
                DbGameStatus.ROUND_IN_PROGRESS,
            }:
                await _schedule_unsupervised_autoplay(game.id)
            payload.append(_serialize_game(game))
        return payload
    finally:
        db.close()


@api.put("/mixed-games/{game_id}")
async def update_mixed_game(
    game_id: int,
    payload: GameUpdatePayload,
    user: Dict[str, Any] = Depends(get_current_user),
):
    db = SyncSessionLocal()
    try:
        game = _get_game_for_user(db, user, game_id)
        if game.status not in {DbGameStatus.CREATED}:
            raise HTTPException(
                status_code=400,
                detail="Games can only be edited while in the CREATED state.",
            )

        config = _coerce_game_config(game)
        update_data = payload.dict(exclude_unset=True)

        if "name" in update_data:
            game.name = update_data["name"]
        if "description" in update_data:
            game.description = update_data["description"]
        if "is_public" in update_data:
            game.is_public = bool(update_data["is_public"])
        if "max_rounds" in update_data:
            game.max_rounds = update_data["max_rounds"]
        if "progression_mode" in update_data:
            config["progression_mode"] = update_data["progression_mode"]
        if payload.demand_pattern is not None:
            config["demand_pattern"] = payload.demand_pattern.dict()
            game.demand_pattern = config["demand_pattern"]
        if payload.pricing_config is not None:
            config["pricing_config"] = payload.pricing_config.dict()
        if payload.node_policies is not None:
            config["node_policies"] = {
                key: policy.dict() for key, policy in payload.node_policies.items()
            }
        if payload.system_config is not None:
            config["system_config"] = payload.system_config
        if payload.global_policy is not None:
            config["global_policy"] = payload.global_policy
        if payload.player_assignments is not None:
            _apply_player_assignments_for_update(db, game, payload.player_assignments, config)

        _ensure_simulation_state(config)
        _save_game_config(db, game, config)
        _touch_game(game)
        db.add(game)
        db.commit()
        db.refresh(game)
        return _serialize_game(game)
    finally:
        db.close()


@api.post("/games/{game_id}/players/{player_id}/orders")
async def submit_order(
    game_id: int,
    player_id: int,
    submission: OrderSubmission,
    user: Dict[str, Any] = Depends(get_current_user),
):
    db = SyncSessionLocal()
    try:
        if submission.quantity < 0:
            raise HTTPException(status_code=400, detail="Quantity must be non-negative")

        game = _get_game_for_user(db, user, game_id)
        player = db.query(Player).filter(Player.id == player_id, Player.game_id == game.id).first()
        if not player:
            raise HTTPException(status_code=404, detail="Player not found for this game")

        config = _coerce_game_config(game)
        round_record = _ensure_round(db, game)

        # Record player action for auditing
        action = (
            db.query(PlayerAction)
            .filter(
                PlayerAction.game_id == game.id,
                PlayerAction.player_id == player.id,
                PlayerAction.round_id == round_record.id,
                PlayerAction.action_type == "order",
            )
            .first()
        )

        timestamp = datetime.utcnow()

        if action:
            action.quantity = submission.quantity
            action.created_at = timestamp
        else:
            action = PlayerAction(
                game_id=game.id,
                round_id=round_record.id,
                player_id=player.id,
                action_type="order",
                quantity=submission.quantity,
                created_at=timestamp,
            )
            db.add(action)

        role_key = str(player.role.value if hasattr(player.role, "value") else player.role).lower()
        pending_snapshot = _pending_orders(config)
        pending_snapshot[role_key] = {
            "player_id": player.id,
            "quantity": submission.quantity,
            "comment": submission.comment,
            "submitted_at": timestamp.isoformat() + "Z",
        }

        _save_game_config(db, game, config)
        db.add(game)
        db.flush()

        auto_advanced = _finalize_round_if_ready(db, game, config, round_record, force=False)
        db.commit()

        progression_mode = _get_progression_mode(game)
        if progression_mode == PROGRESSION_UNSUPERVISED:
            await _schedule_unsupervised_autoplay(game.id)

        return {
            "status": "recorded",
            "auto_advanced": auto_advanced,
            "pending_orders": {
                str(p.role.value if hasattr(p.role, "value") else p.role).lower(): {
                    "player_id": p.id,
                    "quantity": act.quantity,
                }
                for act, p in db.query(PlayerAction, Player)
                .join(Player, Player.id == PlayerAction.player_id)
                .filter(
                    PlayerAction.game_id == game.id,
                    PlayerAction.round_id == round_record.id,
                    PlayerAction.action_type == "order",
                )
            },
            "progression_mode": progression_mode,
        }
    finally:
        db.close()


@api.get("/mixed-games/{game_id}/rounds")
async def list_rounds(game_id: int, user: Dict[str, Any] = Depends(get_current_user)):
    db = SyncSessionLocal()
    try:
        game = _get_game_for_user(db, user, game_id)
        config = _coerce_game_config(game)
        history = config.get("history", [])
        if not history:
            history = _replay_history_from_rounds(db, game)
        players = db.query(Player).filter(Player.game_id == game.id).all()
        players_by_role = {
            str(p.role.value if hasattr(p.role, "value") else p.role).lower(): p
            for p in players
        }

        rounds_payload = []
        for entry in history:
            round_number = entry.get("round")
            player_rounds = []
            for role, player in players_by_role.items():
                order_info = entry.get("orders", {}).get(role, {})
                player_rounds.append(
                    {
                        "player_id": player.id,
                        "role": role.upper(),
                        "order_placed": order_info.get("quantity", 0),
                        "inventory_after": entry.get("inventory_positions", {}).get(role, 0),
                        "backorders_after": entry.get("backlogs", {}).get(role, 0),
                        "comment": order_info.get("comment"),
                    }
                )

            rounds_payload.append(
                {
                    "round_number": round_number,
                    "demand": entry.get("demand", 0),
                    "player_rounds": player_rounds,
                }
            )
        return rounds_payload
    finally:
        db.close()


@api.get("/games/{game_id}/rounds")
async def list_rounds_alias(game_id: int, user: Dict[str, Any] = Depends(get_current_user)):
    return await list_rounds(game_id, user)


@api.get("/mixed-games/{game_id}/rounds/current/status")
async def current_round_status(game_id: int, user: Dict[str, Any] = Depends(get_current_user)):
    db = SyncSessionLocal()
    try:
        game = _get_game_for_user(db, user, game_id)
        config = _coerce_game_config(game)
        pending = _pending_orders(config)
        players = db.query(Player).filter(Player.game_id == game.id).all()
        all_roles = {
            str(p.role.value if hasattr(p.role, "value") else p.role).lower(): p.id
            for p in players
        }
        submitted = {role for role, data in pending.items() if data.get("quantity") is not None}
        outstanding = [role for role in all_roles.keys() if role not in submitted]
        return {
            "game_id": game.id,
            "current_round": game.current_round or 1,
            "progression_mode": _get_progression_mode(game),
            "submitted_roles": list(submitted),
            "outstanding_roles": outstanding,
        }
    finally:
        db.close()


@api.get("/games/{game_id}/rounds/current/status")
async def current_round_status_alias(game_id: int, user: Dict[str, Any] = Depends(get_current_user)):
    return await current_round_status(game_id, user)


@api.get("/mixed-games/{game_id}/report")
async def get_game_report(game_id: int, user: Dict[str, Any] = Depends(get_current_user)):
    db = SyncSessionLocal()
    try:
        game = _get_game_for_user(db, user, game_id)
        return _compute_game_report(db, game)
    finally:
        db.close()


@api.get("/mixed-games/{game_id}/state")
async def get_game_state(game_id: int, user: Dict[str, Any] = Depends(get_current_user)):
    db = SyncSessionLocal()
    try:
        game = _get_game_for_user(db, user, game_id)
        config = _coerce_game_config(game)
        players = db.query(Player).filter(Player.game_id == game.id).all()
        players_payload = [
            {
                "id": player.id,
                "role": player.role.value if hasattr(player.role, "value") else player.role,
                "user_id": player.user_id,
                "is_ai": bool(player.is_ai),
                "ai_strategy": player.ai_strategy,
                "can_see_demand": bool(player.can_see_demand),
            }
            for player in players
        ]
        round_record = _ensure_round(db, game, game.current_round or 1)
        pending = _pending_orders(config)
        history = config.get("history", [])
        if not history:
            history = _replay_history_from_rounds(db, game)
        return {
            "game": _serialize_game(game),
            "progression_mode": _get_progression_mode(game),
            "round": round_record.round_number,
            "pending_orders": pending,
            "history": history,
            "players": players_payload,
        }
    finally:
        db.close()


@api.get("/games/{game_id}")
async def get_game(game_id: int, user: Dict[str, Any] = Depends(get_current_user)):
    db = SyncSessionLocal()
    try:
        game = _get_game_for_user(db, user, game_id)
        payload = _serialize_game(game)
        players = db.query(Player).filter(Player.game_id == game.id).all()
        payload["players"] = [
            {
                "id": player.id,
                "name": player.name,
                "role": player.role.value if hasattr(player.role, "value") else player.role,
                "user_id": player.user_id,
                "is_ai": bool(player.is_ai),
                "ai_strategy": player.ai_strategy,
                "can_see_demand": bool(player.can_see_demand),
            }
            for player in players
        ]
        return payload
    finally:
        db.close()


@api.get("/games/{game_id}/players")
async def list_players(game_id: int, user: Dict[str, Any] = Depends(get_current_user)):
    db = SyncSessionLocal()
    try:
        game = _get_game_for_user(db, user, game_id)
        players = db.query(Player).filter(Player.game_id == game.id).all()
        return [
            {
                "id": player.id,
                "name": player.name,
                "role": player.role.value if hasattr(player.role, "value") else player.role,
                "user_id": player.user_id,
                "is_ai": bool(player.is_ai),
                "ai_strategy": player.ai_strategy,
                "can_see_demand": bool(player.can_see_demand),
            }
            for player in players
        ]
    finally:
        db.close()


@api.post("/mixed-games/{game_id}/start")
async def start_game(
    game_id: int,
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user),
):
    db = SyncSessionLocal()
    try:
        game = _get_game_for_user(db, user, game_id)
        if game.status == DbGameStatus.FINISHED:
            raise HTTPException(status_code=400, detail="Game is already finished")
        payload: Dict[str, Any] = {}
        try:
            if request.headers.get("content-length") not in (None, "0"):
                payload = await request.json()
        except Exception:  # noqa: BLE001
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        debug_requested = _to_bool(
            payload.get("debug_logging")
            or payload.get("debug")
            or payload.get("debug_mode")
        )
        if not debug_requested:
            qp_value = (
                request.query_params.get("debug_logging")
                or request.query_params.get("debug")
                or request.query_params.get("debug_mode")
            )
            if qp_value is not None:
                debug_requested = _to_bool(qp_value)

        config = _coerce_game_config(game)
        config["debug_logging"] = {"enabled": bool(debug_requested)}
        if debug_requested:
            _ensure_debug_log_file(config, game)
        pending = _pending_orders(config)
        pending.clear()
        config.setdefault("history", [])
        _ensure_simulation_state(config)

        if game.current_round is None or game.current_round <= 0:
            game.current_round = 1
        round_record = _ensure_round(db, game, game.current_round)
        round_record.status = "in_progress"
        round_record.started_at = datetime.utcnow()

        game.status = (
            DbGameStatus.ROUND_IN_PROGRESS
            if _get_progression_mode(game) == PROGRESSION_UNSUPERVISED
            else DbGameStatus.STARTED
        )
        _touch_game(game)
        _save_game_config(db, game, config)
        db.add(round_record)
        db.add(game)
        db.commit()
        db.refresh(game)
        payload = _serialize_game(game)
        if _get_progression_mode(game) == PROGRESSION_UNSUPERVISED:
            await _schedule_unsupervised_autoplay(game.id)
        return payload
    finally:
        db.close()


@api.post("/mixed-games/{game_id}/stop")
async def stop_game(game_id: int, user: Dict[str, Any] = Depends(get_current_user)):
    db = SyncSessionLocal()
    try:
        game = _get_game_for_user(db, user, game_id)
        game.status = DbGameStatus.ROUND_COMPLETED
        _touch_game(game)
        db.add(game)
        db.commit()
        db.refresh(game)
        return _serialize_game(game)
    finally:
        db.close()


@api.post("/mixed-games/{game_id}/reset")
async def reset_game(game_id: int, user: Dict[str, Any] = Depends(get_current_user)):
    _cancel_unsupervised_autoplay(game_id)
    db = SyncSessionLocal()
    try:
        game = _get_game_for_user(db, user, game_id)

        # Remove historical round data
        round_ids = [rid for (rid,) in db.query(Round.id).filter(Round.game_id == game.id).all()]
        if round_ids:
            db.query(PlayerAction).filter(PlayerAction.game_id == game.id).delete(synchronize_session=False)
        db.query(Round).filter(Round.game_id == game.id).delete(synchronize_session=False)

        sc_round_ids = [rid for (rid,) in db.query(SupplyGameRound.id).filter(SupplyGameRound.game_id == game.id).all()]
        if sc_round_ids:
            db.query(SupplyPlayerRound).filter(SupplyPlayerRound.round_id.in_(sc_round_ids)).delete(synchronize_session=False)
        db.query(SupplyGameRound).filter(SupplyGameRound.game_id == game.id).delete(synchronize_session=False)
        db.query(SupplyOrder).filter(SupplyOrder.game_id == game.id).delete(synchronize_session=False)

        config = _coerce_game_config(game)
        config["pending_orders"] = {}
        config["history"] = []
        config.pop("simulation_state", None)
        config.pop("engine_state", None)

        node_policies = config.get("node_policies", {})
        sim_params = _simulation_parameters(config)

        players = db.query(Player).filter(Player.game_id == game.id).all()
        for player in players:
            role_key = _role_key(player)
            policy_cfg = MixedGameService._policy_for_node(node_policies, role_key)
            init_inventory = int(policy_cfg.get("init_inventory", sim_params.get("initial_inventory", 12)))
            ship_delay = int(policy_cfg.get("ship_delay", sim_params.get("shipping_lead_time", 2)))
            incoming = [0] * max(1, ship_delay)

            inventory = (
                db.query(PlayerInventory)
                .filter(PlayerInventory.player_id == player.id)
                .first()
            )
            if inventory is None:
                inventory = PlayerInventory(
                    player_id=player.id,
                    current_stock=init_inventory,
                    incoming_shipments=incoming,
                    backorders=0,
                    cost=0.0,
                )
                db.add(inventory)
            else:
                inventory.current_stock = init_inventory
                inventory.incoming_shipments = incoming
                inventory.backorders = 0
                inventory.cost = 0.0

            player.last_order = None
            player.is_ready = False

        _ensure_simulation_state(config)
        _save_game_config(db, game, config)

        game.current_round = 0
        game.status = DbGameStatus.CREATED
        game.started_at = None
        game.finished_at = None
        game.completed_at = None
        _touch_game(game)
        db.add(game)
        db.commit()
        db.refresh(game)
        return _serialize_game(game)
    finally:
        db.close()


@api.post("/mixed-games/{game_id}/next-round")
async def next_round(game_id: int, user: Dict[str, Any] = Depends(get_current_user)):
    db = SyncSessionLocal()
    try:
        game = _get_game_for_user(db, user, game_id)
        if _get_progression_mode(game) == PROGRESSION_UNSUPERVISED:
            raise HTTPException(status_code=400, detail="Unsupervised games advance automatically")

        config = _coerce_game_config(game)
        round_record = _ensure_round(db, game, game.current_round or 1)
        if not _all_players_submitted(db, game, round_record):
            raise HTTPException(status_code=400, detail="All players must submit orders before advancing")

        if not _finalize_round_if_ready(db, game, config, round_record, force=True):
            raise HTTPException(status_code=400, detail="Unable to advance round")

        db.commit()
        db.refresh(game)
        return _serialize_game(game)
    finally:
        db.close()


@api.post("/mixed-games/{game_id}/finish")
async def finish_game(game_id: int, user: Dict[str, Any] = Depends(get_current_user)):
    db = SyncSessionLocal()
    try:
        game = _get_game_for_user(db, user, game_id)
        config = _coerce_game_config(game)
        round_record = _ensure_round(db, game, game.current_round or 1)
        pending = _pending_orders(config)
        if pending and not _finalize_round_if_ready(db, game, config, round_record, force=True):
            _save_game_config(db, game, config)

        game.status = DbGameStatus.FINISHED
        if game.max_rounds:
            game.current_round = max(game.current_round or 0, game.max_rounds)
        _touch_game(game)
        _save_game_config(db, game, config)
        db.add(round_record)
        db.add(game)
        db.commit()
        db.refresh(game)
        return _serialize_game(game)
    finally:
        db.close()


# Simple model status for UI banner
# ------------------------------------------------------------------------------
# Group management
# ------------------------------------------------------------------------------

@api.get("/groups", response_model=List[GroupSchema], tags=["groups"])
def list_groups_endpoint(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_sync_session),
):
    require_system_admin(current_user)
    service = GroupService(db)
    return service.get_groups()


@api.post("/groups/default", response_model=GroupSchema, tags=["groups"])
def ensure_default_group_endpoint(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_sync_session),
):
    require_system_admin(current_user)
    service = GroupService(db)
    groups = service.get_groups()
    if groups:
        return groups[0]
    payload = _default_group_payload()
    return service.create_group(payload)


@api.post("/groups", response_model=GroupSchema, status_code=status.HTTP_201_CREATED, tags=["groups"])
def create_group_endpoint(
    group_in: GroupCreate,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_sync_session),
):
    require_system_admin(current_user)
    service = GroupService(db)
    return service.create_group(group_in)


@api.put("/groups/{group_id}", response_model=GroupSchema, tags=["groups"])
def update_group_endpoint(
    group_id: int,
    group_update: GroupUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_sync_session),
):
    require_system_admin(current_user)
    service = GroupService(db)
    return service.update_group(group_id, group_update)


@api.delete("/groups/{group_id}", tags=["groups"])
def delete_group_endpoint(
    group_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_sync_session),
):
    require_system_admin(current_user)
    service = GroupService(db)
    return service.delete_group(group_id)


_MODEL_STATUS = {
    "is_trained": True,
    "last_modified": datetime.utcnow().isoformat() + "Z",
    "file_size_mb": 12.3,
    "epoch": 10,
    "training_loss": 0.1234,
}


@api.get("/model/status")
async def model_status():
    return _MODEL_STATUS

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
