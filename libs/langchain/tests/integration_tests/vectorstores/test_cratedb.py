"""
Test CrateDB `FLOAT_VECTOR` / `KNN_MATCH` functionality.

cd tests/integration_tests/vectorstores/docker-compose
docker-compose -f cratedb.yml up
"""
import os
from typing import List, Tuple

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from langchain.docstore.document import Document
from langchain.vectorstores.cratedb import BaseModel, CrateDBVectorSearch
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

CONNECTION_STRING = CrateDBVectorSearch.connection_string_from_db_params(
    driver=os.environ.get("TEST_CRATEDB_DRIVER", "crate"),
    host=os.environ.get("TEST_CRATEDB_HOST", "localhost"),
    port=int(os.environ.get("TEST_CRATEDB_PORT", "4200")),
    database=os.environ.get("TEST_CRATEDB_DATABASE", "testdrive"),
    user=os.environ.get("TEST_CRATEDB_USER", "crate"),
    password=os.environ.get("TEST_CRATEDB_PASSWORD", ""),
)


# TODO: Try 1536 after https://github.com/crate/crate/pull/14699.
# ADA_TOKEN_COUNT = 14
ADA_TOKEN_COUNT = 1024
# ADA_TOKEN_COUNT = 1536


@pytest.fixture
def engine() -> sa.Engine:
    """
    Return an SQLAlchemy engine object.
    """
    return sa.create_engine(CONNECTION_STRING, echo=False)


@pytest.fixture(autouse=True)
def drop_tables(engine: sa.Engine) -> None:
    """
    Drop database tables.
    """
    try:
        BaseModel.metadata.drop_all(engine, checkfirst=False)
    except Exception:
        pass


@pytest.fixture
def prune_tables(engine: sa.Engine) -> None:
    """
    Delete data from database tables.
    """
    with engine.connect() as conn:
        with Session(conn) as session:
            from langchain.vectorstores.cratedb import CollectionStore, EmbeddingStore

            session.query(CollectionStore).delete()
            session.query(EmbeddingStore).delete()


def decode_output(
    output: List[Tuple[Document, float]]
) -> Tuple[List[Document], List[float]]:
    """
    Decode a typical API result into separate `documents` and `scores`.
    It is needed as utility function in some test cases to compensate
    for different and/or flaky score values, when compared to the
    original implementation.
    """
    documents = [item[0] for item in output]
    scores = [round(item[1], 1) for item in output]
    return documents, scores


class FakeEmbeddingsWithAdaDimension(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [
            [float(1.0)] * (ADA_TOKEN_COUNT - 1) + [float(i)] for i in range(len(texts))
        ]

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (ADA_TOKEN_COUNT - 1) + [float(0.0)]


def test_cratedb_texts() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = CrateDBVectorSearch.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_cratedb_embeddings() -> None:
    """Test end to end construction with embeddings and search."""
    texts = ["foo", "bar", "baz"]
    text_embeddings = FakeEmbeddingsWithAdaDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = CrateDBVectorSearch.from_embeddings(
        text_embeddings=text_embedding_pairs,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_cratedb_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorSearch.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


def test_cratedb_with_metadatas_with_scores() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorSearch.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score("foo", k=1)
    # TODO: Original:
    #       assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]  # noqa: E501
    assert output in [
        [(Document(page_content="foo", metadata={"page": "0"}), 1.0828735)],
        [(Document(page_content="foo", metadata={"page": "0"}), 1.1307646)],
    ]


def test_cratedb_with_filter_match() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorSearch.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score("foo", k=1, filter={"page": "0"})
    # TODO: Original:
    #       assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]  # noqa: E501
    assert output in [
        [(Document(page_content="foo", metadata={"page": "0"}), 1.2615292)],
        [(Document(page_content="foo", metadata={"page": "0"}), 1.3979403)],
        [(Document(page_content="foo", metadata={"page": "0"}), 1.5065275)],
    ]


def test_cratedb_with_filter_distant_match() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorSearch.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    # TODO: Original:
    # output = docsearch.similarity_search_with_score("foo", k=1, filter={"page": "2"})
    output = docsearch.similarity_search_with_score("foo", k=3, filter={"page": "2"})
    # TODO: Original:
    #       assert output == [
    #         (Document(page_content="baz", metadata={"page": "2"}), 0.0013003906671379406)  # noqa: E501
    #       ]
    documents, scores = decode_output(output)
    assert documents == [
        Document(page_content="baz", metadata={"page": "2"}),
    ]
    assert scores in [
        [0.5],
        [0.6],
        [0.7],
    ]


def test_cratedb_with_filter_no_match() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorSearch.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score("foo", k=1, filter={"page": "5"})
    assert output == []


def test_cratedb_collection_with_metadata() -> None:
    """Test end to end collection construction"""
    cratedb_vector = CrateDBVectorSearch(
        collection_name="test_collection",
        collection_metadata={"foo": "bar"},
        embedding_function=FakeEmbeddingsWithAdaDimension(),
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    session = Session(cratedb_vector.connect())
    collection = cratedb_vector.get_collection(session)
    if collection is None:
        assert False, "Expected a CollectionStore object but received None"
    else:
        assert collection.name == "test_collection"
        assert collection.cmetadata == {"foo": "bar"}


def test_cratedb_with_filter_in_set() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorSearch.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score(
        "foo", k=2, filter={"page": {"IN": ["0", "2"]}}
    )
    # TODO: Original:
    """
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}), 0.0),
        (Document(page_content="baz", metadata={"page": "2"}), 0.0013003906671379406),
    ]
    """
    documents, scores = decode_output(output)
    assert documents == [
        Document(page_content="foo", metadata={"page": "0"}),
        Document(page_content="baz", metadata={"page": "2"}),
    ]
    assert scores == [2.1, 1.3]


def test_cratedb_delete_docs() -> None:
    """Add and delete documents."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorSearch.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        ids=["1", "2", "3"],
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    docsearch.delete(["1", "2"])
    with docsearch._make_session() as session:
        records = list(session.query(docsearch.EmbeddingStore).all())
        # ignoring type error since mypy cannot determine whether
        # the list is sortable
        assert sorted(record.custom_id for record in records) == ["3"]  # type: ignore

    docsearch.delete(["2", "3"])  # Should not raise on missing ids
    with docsearch._make_session() as session:
        records = list(session.query(docsearch.EmbeddingStore).all())
        # ignoring type error since mypy cannot determine whether
        # the list is sortable
        assert sorted(record.custom_id for record in records) == []  # type: ignore


def test_cratedb_relevance_score() -> None:
    """Test to make sure the relevance score is scaled to 0-1."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorSearch.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )

    output = docsearch.similarity_search_with_relevance_scores("foo", k=3)
    """
    # TODO: Original code, where the `distance` is stable.
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}), 1.0),
        (Document(page_content="bar", metadata={"page": "1"}), 0.9996744261675065),
        (Document(page_content="baz", metadata={"page": "2"}), 0.9986996093328621),
    ]
    """
    documents, scores = decode_output(output)
    assert documents == [
        Document(page_content="foo", metadata={"page": "0"}),
        Document(page_content="bar", metadata={"page": "1"}),
        Document(page_content="baz", metadata={"page": "2"}),
    ]
    assert scores == [0.8, 0.4, 0.2]


def test_cratedb_retriever_search_threshold() -> None:
    """Test using retriever for searching with threshold."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorSearch.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )

    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        # TODO: Original:
        #       search_kwargs={"k": 3, "score_threshold": 0.999},
        search_kwargs={"k": 3, "score_threshold": 0.333},
    )
    output = retriever.get_relevant_documents("summer")
    assert output == [
        Document(page_content="foo", metadata={"page": "0"}),
        Document(page_content="bar", metadata={"page": "1"}),
    ]


def test_cratedb_retriever_search_threshold_custom_normalization_fn() -> None:
    """Test searching with threshold and custom normalization function"""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorSearch.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
        relevance_score_fn=lambda d: d * 0,
    )

    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )
    output = retriever.get_relevant_documents("foo")
    assert output == []


def test_cratedb_max_marginal_relevance_search() -> None:
    """Test max marginal relevance search."""
    texts = ["foo", "bar", "baz"]
    docsearch = CrateDBVectorSearch.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.max_marginal_relevance_search("foo", k=1, fetch_k=3)
    assert output == [Document(page_content="foo")]


def test_cratedb_max_marginal_relevance_search_with_score() -> None:
    """Test max marginal relevance search with relevance scores."""
    texts = ["foo", "bar", "baz"]
    docsearch = CrateDBVectorSearch.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.max_marginal_relevance_search_with_score("foo", k=1, fetch_k=3)
    # TODO: Original:
    #       assert output == [(Document(page_content="foo"), 0.0)]
    assert output in [
        [(Document(page_content="foo"), 1.0606961)],
        [(Document(page_content="foo"), 1.0828735)],
        [(Document(page_content="foo"), 1.1307646)],
    ]
