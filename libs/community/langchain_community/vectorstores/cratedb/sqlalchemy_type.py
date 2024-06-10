# TODO: Refactor to CrateDB SQLAlchemy dialect.
import typing as t

import numpy as np
import numpy.typing as npt
import sqlalchemy as sa
from sqlalchemy.types import UserDefinedType

__all__ = ["FloatVector"]


def from_db(value: t.Iterable) -> t.Optional[npt.ArrayLike]:
    # from `pgvector.utils`
    # could be ndarray if already cast by lower-level driver
    if value is None or isinstance(value, np.ndarray):
        return value

    return np.array(value, dtype=np.float32)


def to_db(value: t.Any, dim: t.Optional[int] = None) -> t.Optional[t.List]:
    # from `pgvector.utils`
    if value is None:
        return value

    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError("expected ndim to be 1")

        if not np.issubdtype(value.dtype, np.integer) and not np.issubdtype(
            value.dtype, np.floating
        ):
            raise ValueError("dtype must be numeric")

        value = value.tolist()

    if dim is not None and len(value) != dim:
        raise ValueError("expected %d dimensions, not %d" % (dim, len(value)))

    return value


class FloatVector(UserDefinedType):
    """
    https://cratedb.com/docs/crate/reference/en/latest/general/ddl/data-types.html#float-vector
    https://cratedb.com/docs/crate/reference/en/latest/general/builtins/scalar-functions.html#scalar-knn-match
    """

    cache_ok = True

    def __init__(self, dim: t.Optional[int] = None) -> None:
        super(UserDefinedType, self).__init__()
        self.dim = dim

    def get_col_spec(self, **kw: t.Any) -> str:
        if self.dim is None:
            return "FLOAT_VECTOR"
        return "FLOAT_VECTOR(%d)" % self.dim

    def bind_processor(self, dialect: sa.Dialect) -> t.Callable:
        def process(value: t.Iterable) -> t.Optional[t.List]:
            return to_db(value, self.dim)

        return process

    def result_processor(self, dialect: sa.Dialect, coltype: t.Any) -> t.Callable:
        def process(value: t.Any) -> t.Optional[npt.ArrayLike]:
            return from_db(value)

        return process

    """
    CrateDB currently only supports similarity function `VectorSimilarityFunction.EUCLIDEAN`.
    -- https://github.com/crate/crate/blob/1ca5c6dbb2/server/src/main/java/io/crate/types/FloatVectorType.java#L55
    
    On the other hand, pgvector use a comparator to apply different similarity functions as operators,
    see `pgvector.sqlalchemy.Vector.comparator_factory`.
    
    <->: l2/euclidean_distance
    <#>: max_inner_product
    <=>: cosine_distance
    
    TODO: Discuss.
    """  # noqa: E501
