"""add training artifacts table"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20250923100000"
down_revision = "202409300001_add_llm_model_to_players"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "supply_chain_training_artifacts",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("config_id", sa.Integer(), sa.ForeignKey("supply_chain_configs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("dataset_name", sa.String(length=255), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "ix_supply_chain_training_artifacts_config_id",
        "supply_chain_training_artifacts",
        ["config_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_supply_chain_training_artifacts_config_id", table_name="supply_chain_training_artifacts")
    op.drop_table("supply_chain_training_artifacts")
