from functools import lru_cache
from typing import Optional, Tuple

import sqlalchemy
from crate.client.sqlalchemy.types import ObjectType
from sqlalchemy.orm import Session, relationship

from langchain.vectorstores.cratedb.base import BaseModel
from langchain.vectorstores.cratedb.sqlalchemy_type import FloatVector


@lru_cache
def model_factory(dimensions: int):
    class CollectionStore(BaseModel):
        """Collection store."""

        __tablename__ = "collection"

        name = sqlalchemy.Column(sqlalchemy.String)
        cmetadata: sqlalchemy.Column = sqlalchemy.Column(ObjectType)

        embeddings = relationship(
            "EmbeddingStore",
            back_populates="collection",
            passive_deletes=True,
        )

        @classmethod
        def get_by_name(
            cls, session: Session, name: str
        ) -> Optional["CollectionStore"]:
            try:
                return (
                    session.query(cls).filter(cls.name == name).first()  # type: ignore[attr-defined]  # noqa: E501
                )
            except sqlalchemy.exc.ProgrammingError as ex:
                if "RelationUnknown" not in str(ex):
                    raise
            return None

        @classmethod
        def get_or_create(
            cls,
            session: Session,
            name: str,
            cmetadata: Optional[dict] = None,
        ) -> Tuple["CollectionStore", bool]:
            """
            Get or create a collection.
            Returns [Collection, bool] where the bool is True if the collection was created.
            """
            created = False
            collection = cls.get_by_name(session, name)
            if collection:
                return collection, created

            collection = cls(name=name, cmetadata=cmetadata)
            session.add(collection)
            session.commit()
            created = True
            return collection, created

    class EmbeddingStore(BaseModel):
        """Embedding store."""

        __tablename__ = "embedding"

        collection_id = sqlalchemy.Column(
            sqlalchemy.String,
            sqlalchemy.ForeignKey(
                f"{CollectionStore.__tablename__}.uuid",
                ondelete="CASCADE",
            ),
        )
        collection = relationship("CollectionStore", back_populates="embeddings")

        embedding = sqlalchemy.Column(FloatVector(dimensions))
        document = sqlalchemy.Column(sqlalchemy.String, nullable=True)
        cmetadata: sqlalchemy.Column = sqlalchemy.Column(ObjectType, nullable=True)

        # custom_id : any user defined id
        custom_id = sqlalchemy.Column(sqlalchemy.String, nullable=True)

    return CollectionStore, EmbeddingStore
