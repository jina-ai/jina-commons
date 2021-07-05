# Helper functions for batching documents in executors.
from typing import List, Generator, Optional

from jina import DocumentArray


def _batch_generator(data: DocumentArray, batch_size: int) -> Generator[DocumentArray, None, None]:
    for i in range(0, len(data), batch_size):
        yield data[i: i + batch_size]


def get_docs_batch_generator(
    docs: DocumentArray,
    traversal_path: List[str],
    batch_size: int,
    needs_attr: Optional[str] = None
) -> Generator[DocumentArray, None, None]:
    """
    Creates a `Generator` that yields `DocumentArray` of size `batch_size` until `docs` is fully traversed along
    the `traversal_path`. The None `docs` are filtered out and optionally the `docs` can be filtered by checking for
    the existence of a `Document` attribute.
    Note, that the last batch might be smaller than `batch_size`.

    Example Usage:

    >>> for batch in get_docs_batch_generator(docs, ['r'], 10, needs_attr='blob'):
    >>>     ...

    :param docs: A document array.
    :param traversal_path: Specifies along which "axis" the document shall be traversed.
    :param batch_size: Size of each generated batch (except the last one, which might be smaller)
    :param needs_attr: Optionally, you can filter out docs which don't have this attribute

    :return: Generator
    """
    assert batch_size > 0, 'Batch size must be greater zero.'
    flat_docs = docs.traverse_flat(traversal_path)
    if needs_attr:
        flat_docs = [doc for doc in flat_docs if getattr(doc, needs_attr) is not None]

    filtered_docs = DocumentArray([doc for doc in flat_docs if doc is not None])

    return _batch_generator(filtered_docs, batch_size)
