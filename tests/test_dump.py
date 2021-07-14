from jina import DocumentArray, Document
import numpy as np

from jina_commons.indexers.dump import export_dump_streaming, _doc_without_embedding, dump_docs, import_vectors


def test_dump_dtype(tmpdir):
    docs = DocumentArray()
    d1 = Document(embedding=np.array([1.3, 1.2], dtype=np.float32))
    docs.append(d1)

    dump_docs(docs, tmpdir, 1, dtype=np.float16)

    _, vecs = import_vectors(tmpdir, '0')
    vecs = list(vecs)
    assert vecs[0].dtype == np.float16
