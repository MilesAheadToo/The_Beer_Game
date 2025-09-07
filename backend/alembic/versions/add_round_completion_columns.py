"""add round completion columns

Revision ID: add_round_completion_columns
Revises: update_player_model
Create Date: 2025-09-07 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = 'add_round_completion_columns'
down_revision = 'update_player_model'
branch_labels = None
depends_on = None

def upgrade():
    # Add is_completed column with default False
    op.add_column('game_rounds', 
                 sa.Column('is_completed', 
                          sa.Boolean(), 
                          server_default=sa.text('0'), 
                          nullable=False))
    
    # Add completed_at column, nullable
    op.add_column('game_rounds',
                 sa.Column('completed_at', 
                          sa.DateTime(), 
                          nullable=True))
    
    # Add is_processed column with default False
    op.add_column('game_rounds',
                 sa.Column('is_processed',
                          sa.Boolean(),
                          server_default=sa.text('0'),
                          nullable=False))

def downgrade():
    op.drop_column('game_rounds', 'is_completed')
    op.drop_column('game_rounds', 'completed_at')
    op.drop_column('game_rounds', 'is_processed')
