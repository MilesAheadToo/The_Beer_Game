"""Update Player model for agent/human support

Revision ID: 1234abcd5678
Revises: <previous_migration_id>
Create Date: 2025-09-01 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '1234abcd5678'
down_revision = '<previous_migration_id>'
branch_labels = None
depends_on = None

def upgrade():
    # Add new columns to players table
    op.add_column('players', sa.Column('is_ai', sa.Boolean(), server_default='false', nullable=False))
    op.add_column('players', sa.Column('ai_strategy', sa.String(20), nullable=True))
    op.add_column('players', sa.Column('can_see_demand', sa.Boolean(), server_default='false', nullable=False))
    
    # Add status column to games table
    op.add_column('games', sa.Column('status', sa.String(20), server_default='created', nullable=False))
    
    # Set is_ai based on existing data (all existing players are human)
    op.execute("UPDATE players SET is_ai = false")
    
    # Create index for faster lookups
    op.create_index('idx_players_game_role', 'players', ['game_id', 'role'], unique=True)
    op.create_index('idx_players_game_user', 'players', ['game_id', 'user_id'], unique=False)

def downgrade():
    op.drop_index('idx_players_game_user', table_name='players')
    op.drop_index('idx_players_game_role', table_name='players')
    op.drop_column('players', 'can_see_demand')
    op.drop_column('players', 'ai_strategy')
    op.drop_column('players', 'is_ai')
    op.drop_column('games', 'status')
