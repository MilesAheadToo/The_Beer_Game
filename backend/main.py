# /backend/app/main.py
from datetime import datetime, timedelta
import os
from typing import Optional, Dict, Any, List

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

from sqlalchemy.orm import Session

from app.services.group_service import GroupService
from app.services.bootstrap import build_default_group_payload, ensure_default_group_and_game
from app.schemas.group import GroupCreate, GroupUpdate, Group as GroupSchema
from app.db.session import sync_engine

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
    }
}

# Allow default group admin to sign in using the same lightweight auth shim.
_FAKE_USERS["groupadmin@daybreak.ai"] = {
    "id": 2,
    "email": "groupadmin@daybreak.ai",
    "name": "Group Administrator",
    "role": "groupadmin",
    "passwords": {"Daybreak@2025", "DayBreak@2025"},
    "aliases": {"groupadmin", "defaultadmin"},
    "group_id": 1,
    "is_superuser": False,
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
    """Accept email or simple alias (e.g., 'admin')."""
    # Try direct email key first
    user = _FAKE_USERS.get(username)
    if not user:
        # Try lookup by alias
        for u in _FAKE_USERS.values():
            if username.lower() in {a.lower() for a in u.get("aliases", set())}:
                user = u
                break
    if not user:
        return None
    # Accept any of the allowed dev passwords for convenience
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

# Alias to match frontend expectation '/auth/refresh-token'
@api.post("/auth/refresh-token", response_model=TokenResponse, tags=["auth"])
async def refresh_alias(response: Response, refresh_cookie: Optional[str] = Cookie(default=None, alias=REFRESH_COOKIE_NAME)):
    return await refresh(response=response, refresh_cookie=refresh_cookie)

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
    type: str  # supplier|manufacturer|distributor|retailer
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
    supplier_lead_times: Mapping[str, float] = Field(default_factory=dict)


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
            Site(id="supplier_1", type="supplier", name="Supplier 1", items_sold=["item_1"]),
            Site(id="manufacturer_1", type="manufacturer", name="Manufacturer 1", items_sold=["item_1"]),
            Site(id="distributor_1", type="distributor", name="Distributor 1", items_sold=["item_1"]),
            Site(id="retailer_1", type="retailer", name="Retailer 1", items_sold=["item_1"]),
        ],
        site_item_settings={
            "supplier_1": {"item_1": SiteItemSettings(inventory_target=20, holding_cost=0.5, backorder_cost=1.0, avg_selling_price=7.0, standard_cost=5.0, moq=0)},
            "manufacturer_1": {"item_1": SiteItemSettings(inventory_target=20, holding_cost=0.5, backorder_cost=1.0, avg_selling_price=7.0, standard_cost=5.0, moq=0)},
            "distributor_1": {"item_1": SiteItemSettings(inventory_target=20, holding_cost=0.5, backorder_cost=1.0, avg_selling_price=7.0, standard_cost=5.0, moq=0)},
            "retailer_1": {"item_1": SiteItemSettings(inventory_target=20, holding_cost=0.5, backorder_cost=1.0, avg_selling_price=7.0, standard_cost=5.0, moq=0)},
        },
        lanes=[
            Lane(from_site_id="supplier_1", to_site_id="manufacturer_1", item_id="item_1", lead_time=2, capacity=None, otif_target=0.95),
            Lane(from_site_id="manufacturer_1", to_site_id="distributor_1", item_id="item_1", lead_time=2, capacity=None, otif_target=0.95),
            Lane(from_site_id="distributor_1", to_site_id="retailer_1", item_id="item_1", lead_time=2, capacity=None, otif_target=0.95),
        ],
        retailer_demand=RetailerDemand(distribution="profile", params={"week1_4": 4, "week5_plus": 8}, expected_delivery_offset=1),
        supplier_lead_times={"item_1": 2},
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


def _serialize_game(game: DbGame) -> Dict[str, Any]:
    if isinstance(game.status, DbGameStatus):
        status = STATUS_REMAPPING.get(game.status, game.status.value)
    else:
        status = str(game.status or "")
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
        "config": game.config or {},
        "demand_pattern": game.demand_pattern or {},
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
        return [_serialize_game(game) for game in games]
    finally:
        db.close()


@api.post("/mixed-games/{game_id}/start")
async def start_game(game_id: int, user: Dict[str, Any] = Depends(get_current_user)):
    db = SyncSessionLocal()
    try:
        game = _get_game_for_user(db, user, game_id)
        if game.status == DbGameStatus.FINISHED:
            raise HTTPException(status_code=400, detail="Game is already finished")
        if game.current_round is None or game.current_round <= 0:
            game.current_round = 1
        game.status = DbGameStatus.STARTED
        _touch_game(game)
        db.add(game)
        db.commit()
        db.refresh(game)
        return _serialize_game(game)
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


@api.post("/mixed-games/{game_id}/next-round")
async def next_round(game_id: int, user: Dict[str, Any] = Depends(get_current_user)):
    db = SyncSessionLocal()
    try:
        game = _get_game_for_user(db, user, game_id)
        current = game.current_round or 0
        limit = game.max_rounds or current
        if current < limit:
            game.current_round = current + 1
        game.status = DbGameStatus.ROUND_IN_PROGRESS
        _touch_game(game)
        db.add(game)
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
        game.status = DbGameStatus.FINISHED
        if game.max_rounds:
            game.current_round = max(game.current_round or 0, game.max_rounds)
        _touch_game(game)
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
