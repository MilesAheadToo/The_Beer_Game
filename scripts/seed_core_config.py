"""Provide a minimal seed_core_config stub for database initialisation."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def seed_core_config(*args: Any, **kwargs: Any) -> None:
    """Placeholder implementation to satisfy init_db imports."""

    logger.info("seed_core_config stub invoked; default supply chain seeding handled elsewhere.")
