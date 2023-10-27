from __future__ import annotations

import enum
import math
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

import sqlalchemy
from cratedb_toolkit.sqlalchemy.patch import patch_inspector
from cratedb_toolkit.sqlalchemy.polyfill import (
    polyfill_refresh_after_dml,
    refresh_table,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores.pgvector import PGVector


class DistanceStrategy(str, enum.Enum):
    """Enumerator of the Distance strategies."""

    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    MAX_INNER_PRODUCT = "inner"


DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.EUCLIDEAN

Base = declarative_base()  # type: Any
# Base = declarative_base(metadata=MetaData(schema="langchain"))  # type: Any

_LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"


def generate_uuid() -> str:
    return str(uuid.uuid4())


class BaseModel(Base):
    """Base model for the SQL stores."""

    __abstract__ = True
    uuid = sqlalchemy.Column(sqlalchemy.String, primary_key=True, default=generate_uuid)


def _results_to_docs(docs_and_scores: Any) -> List[Document]:
    """Return docs from docs and scores."""
    return [doc for doc, _ in docs_and_scores]


class CrateDBVectorSearch(PGVector):
    """`CrateDB` vector store.

    To use it, you should have the ``crate[sqlalchemy]`` python package installed.

    Args:
        connection_string: Database connection string.
        embedding_function: Any embedding function implementing
            `langchain.embeddings.base.Embeddings` interface.
        collection_name: The name of the collection to use. (default: langchain)
            NOTE: This is not the name of the table, but the name of the collection.
            The tables will be created when initializing the store (if not exists)
            So, make sure the user has the right permissions to create tables.
        distance_strategy: The distance strategy to use. (default: EUCLIDEAN)
        pre_delete_collection: If True, will delete the collection if it exists.
            (default: False). Useful for testing.

    Example:
        .. code-block:: python

            from langchain.vectorstores import CrateDBVectorSearch
            from langchain.embeddings.openai import OpenAIEmbeddings

            CONNECTION_STRING = "crate://crate@localhost:4200/test3"
            COLLECTION_NAME = "state_of_the_union_test"
            embeddings = OpenAIEmbeddings()
            vectorestore = CrateDBVectorSearch.from_documents(
                embedding=embeddings,
                documents=docs,
                collection_name=COLLECTION_NAME,
                connection_string=CONNECTION_STRING,
            )


    """

    def __post_init__(
        self,
    ) -> None:
        """
        Initialize the store.
        """

        # FIXME: Could be a bug in CrateDB SQLAlchemy dialect.
        patch_inspector()

        self._engine = self.create_engine()
        self.Session = sessionmaker(self._engine)

        # TODO: See what can be improved here.
        polyfill_refresh_after_dml(self.Session)

        # Need to defer initialization, because dimension size
        # can only be figured out at runtime.
        self.CollectionStore = None
        self.EmbeddingStore = None

    def _init_models(self, embedding: List[float]):
        """
        Create SQLAlchemy models at runtime, when not established yet.
        """
        if self.CollectionStore is not None and self.EmbeddingStore is not None:
            return

        size = len(embedding)
        self._init_models_with_dimensionality(size=size)

    def _init_models_with_dimensionality(self, size: int):
        from langchain.vectorstores.cratedb.model import model_factory

        self.CollectionStore, self.EmbeddingStore = model_factory(dimensions=size)

    def get_collection(
        self, session: sqlalchemy.orm.Session
    ) -> Optional["CollectionStore"]:
        if self.CollectionStore is None:
            raise RuntimeError(
                "Collection can't be accessed without specifying dimension size of embedding vectors"
            )
        return self.CollectionStore.get_by_name(session, self.collection_name)

    def add_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add embeddings to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            embeddings: List of list of embedding vectors.
            metadatas: List of metadatas associated with the texts.
            kwargs: vectorstore specific parameters
        """
        if not embeddings:
            return []
        self._init_models(embeddings[0])
        if self.pre_delete_collection:
            self.delete_collection()
        self.create_tables_if_not_exists()
        self.create_collection()
        return super().add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

    def create_tables_if_not_exists(self) -> None:
        """
        Need to overwrite because `Base` is different from upstream.
        """
        Base.metadata.create_all(self._engine)

    def drop_tables(self) -> None:
        """
        Need to overwrite because `Base` is different from upstream.
        """
        Base.metadata.drop_all(self._engine)

    def delete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Delete vectors by ids or uuids.

        Remark: Specialized for CrateDB to synchronize data.

        Args:
            ids: List of ids to delete.

        Remark: Patch for CrateDB needs to overwrite this, in order to
                add a "REFRESH TABLE" statement afterwards. The other
                patch, listening to `after_delete` events seems not be
                able to catch it.
        """
        super().delete(ids=ids, **kwargs)

        # CrateDB: Synchronize data because `on_flush` does not catch it.
        with self.Session() as session:
            refresh_table(session, self.EmbeddingStore)

    @property
    def distance_strategy(self) -> Any:
        if self._distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self.EmbeddingStore.embedding.euclidean_distance
        elif self._distance_strategy == DistanceStrategy.COSINE:
            raise NotImplementedError("Cosine similarity not implemented yet")
        elif self._distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            raise NotImplementedError("Dot-product similarity not implemented yet")
        else:
            raise ValueError(
                f"Got unexpected value for distance: {self._distance_strategy}. "
                f"Should be one of {', '.join([ds.value for ds in DistanceStrategy])}."
            )

    def _results_to_docs_and_scores(self, results: Any) -> List[Tuple[Document, float]]:
        """Return docs and scores from results."""
        docs = [
            (
                Document(
                    page_content=result.EmbeddingStore.document,
                    metadata=result.EmbeddingStore.cmetadata,
                ),
                result._score if self.embedding_function is not None else None,
            )
            for result in results
        ]
        return docs

    def _query_collection(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
    ) -> List[Any]:
        """Query the collection."""
        self._init_models(embedding)
        with self.Session() as session:
            collection = self.get_collection(session)
            if not collection:
                raise ValueError("Collection not found")

            filter_by = self.EmbeddingStore.collection_id == collection.uuid

            if filter is not None:
                filter_clauses = []
                for key, value in filter.items():
                    IN = "in"
                    if isinstance(value, dict) and IN in map(str.lower, value):
                        value_case_insensitive = {
                            k.lower(): v for k, v in value.items()
                        }
                        filter_by_metadata = self.EmbeddingStore.cmetadata[key].in_(
                            value_case_insensitive[IN]
                        )
                        filter_clauses.append(filter_by_metadata)
                    else:
                        filter_by_metadata = self.EmbeddingStore.cmetadata[key] == str(
                            value
                        )  # type: ignore[assignment]
                        filter_clauses.append(filter_by_metadata)

                filter_by = sqlalchemy.and_(filter_by, *filter_clauses)

            _type = self.EmbeddingStore

            results: List[Any] = (
                session.query(  # type: ignore[attr-defined]
                    self.EmbeddingStore,
                    # TODO: Original pgvector code uses `self.distance_strategy`.
                    #       CrateDB currently only supports EUCLIDEAN.
                    #       self.distance_strategy(embedding).label("distance")
                    sqlalchemy.literal_column(
                        f"{self.EmbeddingStore.__tablename__}._score"
                    ).label("_score"),
                )
                .filter(filter_by)
                # CrateDB applies `KNN_MATCH` within the `WHERE` clause.
                .filter(
                    sqlalchemy.func.knn_match(
                        self.EmbeddingStore.embedding, embedding, k
                    )
                )
                .order_by(sqlalchemy.desc("_score"))
                .join(
                    self.CollectionStore,
                    self.EmbeddingStore.collection_id == self.CollectionStore.uuid,
                )
                .limit(k)
            )
        return results

    @classmethod
    def from_texts(  # type: ignore[override]
        cls: Type[CrateDBVectorSearch],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> CrateDBVectorSearch:
        """
        Return VectorStore initialized from texts and embeddings.
        Database connection string is required.

        Either pass it as a parameter, or set the CRATEDB_CONNECTION_STRING
        environment variable.

        Remark: Needs to be overwritten, because CrateDB uses a different
                DEFAULT_DISTANCE_STRATEGY.
        """
        return super().from_texts(  # type: ignore[return-value]
            texts,
            embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            distance_strategy=distance_strategy,  # type: ignore[arg-type]
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

    @classmethod
    def get_connection_string(cls, kwargs: Dict[str, Any]) -> str:
        connection_string: str = get_from_dict_or_env(
            data=kwargs,
            key="connection_string",
            env_key="CRATEDB_CONNECTION_STRING",
        )

        if not connection_string:
            raise ValueError(
                "Database connection string is required."
                "Either pass it as a parameter, or set the "
                "CRATEDB_CONNECTION_STRING environment variable."
            )

        return connection_string

    @classmethod
    def connection_string_from_db_params(
        cls,
        driver: str,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
    ) -> str:
        """Return connection string from database parameters."""
        return str(
            sqlalchemy.URL.create(
                drivername=driver,
                host=host,
                port=port,
                username=user,
                password=password,
                query={"schema": database},
            )
        )

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.
        """
        if self.override_relevance_score_fn is not None:
            return self.override_relevance_score_fn

        # Default strategy is to rely on distance strategy provided
        # in vectorstore constructor
        if self._distance_strategy == DistanceStrategy.COSINE:
            return self._cosine_relevance_score_fn
        elif self._distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self._euclidean_relevance_score_fn
        elif self._distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            return self._max_inner_product_relevance_score_fn
        else:
            raise ValueError(
                "No supported normalization function for distance_strategy of "
                "{self._distance_strategy}. Consider providing relevance_score_fn to "
                "CrateDBVectorSearch constructor."
            )

    @staticmethod
    def _euclidean_relevance_score_fn(score: float) -> float:
        """Return a similarity score on a scale [0, 1]."""
        # The 'correct' relevance function
        # may differ depending on a few things, including:
        # - the distance / similarity metric used by the VectorStore
        # - the scale of your embeddings (OpenAI's are unit normed. Many
        #  others are not!)
        # - embedding dimensionality
        # - etc.
        # This function converts the euclidean norm of normalized embeddings
        # (0 is most similar, sqrt(2) most dissimilar)
        # to a similarity function (0 to 1)

        # Original:
        # return 1.0 - distance / math.sqrt(2)
        return score / math.sqrt(2)
