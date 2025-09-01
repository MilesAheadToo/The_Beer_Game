"""Fix migration to handle existing tables

Revision ID: 002
Revises: 001
Create Date: 2023-10-01 12:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None

def upgrade():
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    existing_tables = inspector.get_table_names()
    
    # Check and create users table if it doesn't exist
    if 'users' not in existing_tables:
        op.create_table(
            'users',
            sa.Column('id', sa.Integer(), nullable=False, primary_key=True, autoincrement=True, index=True),
            sa.Column('username', sa.String(50), unique=True, nullable=False, index=True),
            sa.Column('email', sa.String(100), unique=True, nullable=False, index=True),
            sa.Column('hashed_password', sa.String(100), nullable=False),
            sa.Column('full_name', sa.String(100), nullable=True),
            sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
            sa.Column('is_superuser', sa.Boolean(), default=False, nullable=False),
            sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
            sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP'), nullable=False),
            mysql_engine='InnoDB',
            mysql_charset='utf8mb4',
            mysql_collate='utf8mb4_unicode_ci'
        )
    
    # Check and create refresh_tokens table if it doesn't exist
    if 'refresh_tokens' not in existing_tables:
        op.create_table(
            'refresh_tokens',
            sa.Column('id', sa.Integer(), nullable=False, primary_key=True, autoincrement=True, index=True),
            sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
            sa.Column('token', sa.String(500), unique=True, nullable=False, index=True),
            sa.Column('expires_at', sa.DateTime(), nullable=False),
            sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
            mysql_engine='InnoDB',
            mysql_charset='utf8mb4',
            mysql_collate='utf8mb4_unicode_ci'
        )
    
    # Check and create user_games table if it doesn't exist
    if 'user_games' not in existing_tables:
        # First make sure the games table exists
        if 'games' in existing_tables:
            op.create_table(
                'user_games',
                sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), primary_key=True, index=True),
                sa.Column('game_id', sa.Integer(), sa.ForeignKey('games.id', ondelete='CASCADE'), primary_key=True, index=True),
                sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
                mysql_engine='InnoDB',
                mysql_charset='utf8mb4',
                mysql_collate='utf8mb4_unicode_ci'
            )
    
    # Add user_id column to players table if it doesn't exist
    if 'players' in existing_tables:
        columns = [col['name'] for col in inspector.get_columns('players')]
        if 'user_id' not in columns:
            op.add_column('players', sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True, index=True))

def downgrade():
    # This is a forward-only migration, no downgrade needed
    pass
