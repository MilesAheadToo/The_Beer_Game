"""Add created_by to games and supply_chain_configs; add metadata columns to games

Revision ID: 20240911150000
Revises: 20240910152000
Create Date: 2025-09-11 15:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import text

# revision identifiers, used by Alembic.
revision = '20240911150000'
down_revision = '20240910152000'
branch_labels = None
depends_on = None


def upgrade():
    # games: created_by, is_public, description, demand_pattern
    with op.batch_alter_table('games') as batch_op:
        if not has_column('games', 'created_by'):
            batch_op.add_column(sa.Column('created_by', sa.Integer(), nullable=True))
        if not has_fk('games', 'fk_games_created_by_users'):
            batch_op.create_foreign_key('fk_games_created_by_users', 'users', ['created_by'], ['id'], ondelete='SET NULL')
        if not has_column('games', 'is_public'):
            batch_op.add_column(sa.Column('is_public', sa.Boolean(), server_default=sa.text('0'), nullable=False))
        if not has_column('games', 'description'):
            batch_op.add_column(sa.Column('description', sa.String(length=500), nullable=True))
        if not has_column('games', 'demand_pattern'):
            batch_op.add_column(sa.Column('demand_pattern', sa.JSON(), nullable=True))

    # supply_chain_configs: created_by
    with op.batch_alter_table('supply_chain_configs') as batch_op:
        if not has_column('supply_chain_configs', 'created_by'):
            batch_op.add_column(sa.Column('created_by', sa.Integer(), nullable=True))
        if not has_fk('supply_chain_configs', 'fk_scc_created_by_users'):
            batch_op.create_foreign_key('fk_scc_created_by_users', 'users', ['created_by'], ['id'], ondelete='SET NULL')

    # backfill created_by with admin user (first superuser)
    conn = op.get_bind()
    try:
        admin_id = conn.execute(text("SELECT id FROM users WHERE is_superuser = 1 ORDER BY id ASC LIMIT 1")).scalar()
        if admin_id:
            conn.execute(text("UPDATE games SET created_by = :uid WHERE created_by IS NULL"), {"uid": admin_id})
            conn.execute(text("UPDATE supply_chain_configs SET created_by = :uid WHERE created_by IS NULL"), {"uid": admin_id})
    except Exception:
        # Non-fatal, continue
        pass


def downgrade():
    with op.batch_alter_table('supply_chain_configs') as batch_op:
        if has_fk('supply_chain_configs', 'fk_scc_created_by_users'):
            batch_op.drop_constraint('fk_scc_created_by_users', type_='foreignkey')
        if has_column('supply_chain_configs', 'created_by'):
            batch_op.drop_column('created_by')

    with op.batch_alter_table('games') as batch_op:
        if has_fk('games', 'fk_games_created_by_users'):
            batch_op.drop_constraint('fk_games_created_by_users', type_='foreignkey')
        if has_column('games', 'created_by'):
            batch_op.drop_column('created_by')
        if has_column('games', 'demand_pattern'):
            batch_op.drop_column('demand_pattern')
        if has_column('games', 'description'):
            batch_op.drop_column('description')
        if has_column('games', 'is_public'):
            batch_op.drop_column('is_public')


# Helpers (Alembic runtime has no metadata reflection convenience in this repo)
def has_column(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    res = bind.exec_driver_sql(
        text(
            """
            SELECT COUNT(*) FROM information_schema.COLUMNS 
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t AND COLUMN_NAME = :c
            """
        ),
        {"t": table_name, "c": column_name},
    ).scalar()
    return bool(res)


def has_fk(table_name: str, fk_name: str) -> bool:
    bind = op.get_bind()
    res = bind.exec_driver_sql(
        text(
            """
            SELECT COUNT(*) FROM information_schema.TABLE_CONSTRAINTS 
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t AND CONSTRAINT_NAME = :n AND CONSTRAINT_TYPE = 'FOREIGN KEY'
            """
        ),
        {"t": table_name, "n": fk_name},
    ).scalar()
    return bool(res)

