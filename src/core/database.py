from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pathlib import Path
from typing import List, Dict, Any
from src.core.model import PointModel
from src.core.base import Base


class MetadataDB:
    def __init__(self, collection_path: str):
        self.db_path = Path(collection_path) / "metadata.db"
        self.engine = create_engine(
            f"sqlite:///{self.db_path}?check_same_thread=False",
            connect_args={'timeout': 30}
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def upsert_points(self, points: List[Dict[str, Any]]) -> List[int]:
        session = self.Session()
        new_internal_ids = []
        try:
            for p in points:
                session.query(PointModel).filter_by(
                    external_id=p['external_id'], deleted=False
                ).update({"deleted": True}, synchronize_session=False)

                point = PointModel(
                    external_id=p['external_id'],
                    text=p['text'],
                    payload=p.get('payload', {}),
                    deleted=False,
                )
                session.add(point)
                session.flush()  
                new_internal_ids.append(point.id)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
        return new_internal_ids

    def get_points(self, ids: List[int]) -> List[Dict[str, Any]]:
        if not ids:
            return []
        session = self.Session()
        try:
            points = session.query(PointModel).filter(PointModel.id.in_(ids)).all()
            return [
                {
                    'id': p.id,
                    'external_id': p.external_id,
                    'text': p.text,
                    'payload': p.payload,
                    'deleted': p.deleted,
                }
                for p in points
            ]
        finally:
            session.close()

    def delete_points(self, external_ids: List[int]) -> int:
        if not external_ids:
            return 0
        session = self.Session()
        try:
            result = session.query(PointModel).filter(
                PointModel.external_id.in_(external_ids),
                PointModel.deleted.is_(False),
            ).update({"deleted": True}, synchronize_session=False)
            session.commit()
            return result
        finally:
            session.close()

    def get_active_points(self) -> List[Dict[str, Any]]:
        session = self.Session()
        try:
            points = session.query(PointModel).filter_by(deleted=False).all()
            return [
                {'id': p.id, 'external_id': p.external_id, 'text': p.text, 'payload': p.payload}
                for p in points
            ]
        finally:
            session.close()

    def get_points_count(self) -> int:
        session = self.Session()
        try:
            return session.query(PointModel).filter_by(deleted=False).count()
        finally:
            session.close()