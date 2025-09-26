"""Ensure game_rounds has timing columns expected by services

Revision ID: 20250226090500
Revises: 20250226090000
Create Date: 2025-09-26 09:05:00.000000

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "20250226090500"
down_revision = "20250226090000"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE game_rounds
        ADD COLUMN IF NOT EXISTS started_at DATETIME NULL,
        ADD COLUMN IF NOT EXISTS ended_at DATETIME NULL
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE game_rounds
        DROP COLUMN IF EXISTS ended_at,
        DROP COLUMN IF EXISTS started_at
        """
    )
