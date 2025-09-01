"""Initial migration.

Revision ID: 001
Revises: 
Create Date: 2023-10-01 12:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('username', sa.String(50), unique=True, nullable=False, index=True),
        sa.Column('email', sa.String(100), unique=True, nullable=False, index=True),
        sa.Column('hashed_password', sa.String(100), nullable=False),
        sa.Column('full_name', sa.String(100), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('is_superuser', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False)
    )
    
    # Create refresh_tokens table
    op.create_table(
        'refresh_tokens',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('token', sa.String(500), unique=True, nullable=False, index=True),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False)
    )
    
    # Create games table first if it doesn't exist
    op.create_table(
        'games',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('status', sa.String(20), default='waiting', nullable=False),
        sa.Column('current_round', sa.Integer(), default=0, nullable=False),
        sa.Column('max_rounds', sa.Integer(), default=52, nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False)
    )
    
    # Create user_games association table
    op.create_table(
        'user_games',
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('game_id', sa.Integer(), sa.ForeignKey('games.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False)
    )
    
    # Create players table if it doesn't exist
    op.create_table(
        'players',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('game_id', sa.Integer(), sa.ForeignKey('games.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True, index=True),
        sa.Column('role', sa.String(50), nullable=False),
        sa.Column('inventory', sa.Integer(), default=12, nullable=False),
        sa.Column('backlog', sa.Integer(), default=0, nullable=False),
        sa.Column('last_order', sa.Integer(), default=0, nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False)
    )

def downgrade():
    # Drop all tables in reverse order
    op.drop_table('players')
    op.drop_table('user_games')
    op.drop_table('games')
    op.drop_table('refresh_tokens')
    op.drop_table('users')
