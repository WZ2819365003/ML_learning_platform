"""Create the complete ORM schema baseline.

Revision ID: 0001
Revises:
Create Date: 2026-07-21
"""

from typing import Sequence, Union

from alembic import op

from app.models.database import Base


revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    pass
