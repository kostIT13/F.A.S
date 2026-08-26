from src.core.base import Base
from sqlalchemy import Column, Integer, Boolean, Text, JSON


class PointModel(Base):
    __tablename__ = 'points'

    id = Column(Integer, primary_key=True, autoincrement=True)
    external_id = Column(Integer, nullable=False, index=True)
    text = Column(Text, nullable=False)
    payload = Column(JSON, default={})
    deleted = Column(Boolean, default=False)