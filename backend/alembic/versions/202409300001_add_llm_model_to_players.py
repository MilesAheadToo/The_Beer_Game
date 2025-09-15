"""add llm_model column to players"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '202409300001'
down_revision = '1234abcd5678'
branch_labels = None
depends_on = None

def upgrade():
    op.add_column(
        'players',
        sa.Column('llm_model', sa.String(length=100), server_default='gpt-4o-mini', nullable=True)
    )


def downgrade():
    op.drop_column('players', 'llm_model')
