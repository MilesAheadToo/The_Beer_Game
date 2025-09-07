"""Add round timing columns

Revision ID: add_round_timing_columns
Revises: add_round_completion_columns
Create Date: 2025-09-07 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = 'add_round_timing_columns'
down_revision = 'add_round_completion_columns'
branch_labels = None
depends_on = None

def upgrade():
    # Add round timing columns to games table
    op.add_column('games', 
                 sa.Column('round_time_limit', 
                          sa.Integer(), 
                          server_default='60', 
                          nullable=False))
    
    op.add_column('games',
                 sa.Column('current_round_ends_at', 
                          sa.DateTime(), 
                          nullable=True))
    
    # Add is_processed column to game_rounds
    op.add_column('game_rounds',
                 sa.Column('is_processed',
                          sa.Boolean(),
                          server_default=sa.text('0'),
                          nullable=False))

def downgrade():
    op.drop_column('games', 'round_time_limit')
    op.drop_column('games', 'current_round_ends_at')
    op.drop_column('game_rounds', 'is_processed')
