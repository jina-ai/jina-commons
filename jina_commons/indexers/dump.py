import os
import sys
import urllib
from typing import Tuple, Generator, BinaryIO, TextIO, Union, List

import numpy as np
from jina import DocumentArray, Document
from jina.logging.logger import JinaLogger
from psycopg2 import connect

BYTE_PADDING = 4
DUMP_DTYPE = np.float64

logger = JinaLogger(__name__)


def _doc_without_embedding(d):
    new_doc = Document(d, copy=True)
    new_doc.ClearField('embedding')
    new_doc.update_content_hash()
    return new_doc.SerializeToString()


def _generator_from_docs(docs):
    for doc in docs:
        yield doc.id, doc.embedding, _doc_without_embedding(doc)


def dump_docs(docs: Union[DocumentArray, List[Document]], path: str, shards: int):
    """dump a DocumentArray"""
    export_dump_streaming(path, shards, len(docs), _generator_from_docs(docs))


def export_dump_streaming(
        path: str,
        shards: int,
        size: int,
        data: Generator[Tuple[str, np.array, bytes], None, None],
):
    """Export the data to a path, based on sharding,

    :param path: path to dump
    :param shards: the nr of shards this pea is part of
    :param size: total amount of entries
    :param data: the generator of the data (ids, vectors, metadata)
    """
    logger.info(f'Dumping {size} docs to {path} for {shards} shards')
    _handle_dump(data, path, shards, size)


def _handle_dump(
        data: Generator[Tuple[str, np.array, bytes], None, None],
        path: str,
        shards: int,
        size: int,
):
    if not os.path.exists(path):
        os.makedirs(path)

    # directory must be empty to be safe
    if not os.listdir(path):
        size_per_shard = size // shards
        extra = size % shards
        shard_range = list(range(shards))
        for shard_id in shard_range:
            if shard_id == shard_range[-1]:
                size_this_shard = size_per_shard + extra
            else:
                size_this_shard = size_per_shard
            _write_shard_data(data, path, shard_id, size_this_shard)
    else:
        raise Exception(
            f'path for dump {path} contains data. Please empty. Not dumping...'
        )


def _write_shard_data(
        data: Generator[Tuple[str, np.array, bytes], None, None],
        path: str,
        shard_id: int,
        size_this_shard: int,
):
    shard_path = os.path.join(path, str(shard_id))
    shard_docs_written = 0
    os.makedirs(shard_path)
    vectors_fp, metas_fp, ids_fp = _get_file_paths(shard_path)
    with open(vectors_fp, 'wb') as vectors_fh, open(metas_fp, 'wb') as metas_fh, open(
            ids_fp, 'w'
    ) as ids_fh:
        while shard_docs_written < size_this_shard:
            _write_shard_files(data, ids_fh, metas_fh, vectors_fh)
            shard_docs_written += 1


def _write_shard_files(
        data: Generator[Tuple[str, np.array, bytes], None, None],
        ids_fh: TextIO,
        metas_fh: BinaryIO,
        vectors_fh: BinaryIO,
):
    id_, vec, meta = next(data)
    # need to ensure compatibility to read time
    if vec is not None:
        vec = vec.astype(DUMP_DTYPE)
        vec_bytes = vec.tobytes()
        vectors_fh.write(len(vec_bytes).to_bytes(BYTE_PADDING, sys.byteorder) + vec_bytes)
    metas_fh.write(len(meta).to_bytes(BYTE_PADDING, sys.byteorder) + meta)
    ids_fh.write(id_ + '\n')


def _import_vectors_psql(path, shard_ids):
    # see https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING
    connection_string = urllib.parse.urlparse(path)
    if connection_string.hostname is None or \
            connection_string.port is None or \
            connection_string.username is None or \
            connection_string.password is None or \
            connection_string.path is None or \
            'table=' not in connection_string.query:
        raise ValueError(
            'Missing one required parameter in postgres path. Must be like: "postgres://username:password@host:port/dbname?table=table_name"')
    connection = connect(
        host=connection_string.hostname,
        port=connection_string.port,
        user=connection_string.username,
        password=connection_string.password,
        dbname=connection_string.path.strip('/')
    )
    cursor = connection.cursor()
    cursor.execute(
        f'SELECT ID, DOC FROM {connection_string.query.split("table=")[1]} WHERE SHARD IN (",".join(shard_ids))'
    )
    records = cursor.fetchall()
    for rec in records:
        doc = Document(bytes(rec[1]))
        vec = doc.embedding
        metas = doc_without_embedding(doc)
        yield rec[0], vec, metas


def _is_psql(path):
    # see https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING
    connection_string = urllib.parse.urlparse(path)
    return connection_string.scheme == 'postgres' or connection_string.scheme == 'postgresql'


def import_vectors(path: str, pea_id: str = None, shard_ids: List[str] = None):
    """Import id and vectors

    :param path: the path to the dump
    :param pea_id: the id of the pea (as part of the shards)
    :param shard_ids: the ids of the shards to obtain (in PSQL
    :return: the generators for the ids and for the vectors
    """
    # TODO check if str is PSQL connection string
    # TODO maybe pea_id is pea_ids ?
    # else below:
    if _is_psql(path) and isinstance(shard_ids, list) and isinstance(shard_ids[0], str):
        return _import_vectors_psql(path, shard_ids)
    else:
        logger.info(f'Importing ids and vectors from {path} for pea_id {pea_id}')
        path = os.path.join(path, pea_id)
        ids_gen = _ids_gen(path)
        vecs_gen = _vecs_gen(path)
        return ids_gen, vecs_gen


def import_metas(path: str, pea_id: str):
    """Import id and metadata

    :param path: the path of the dump
    :param pea_id: the id of the pea (as part of the shards)
    :return: the generators for the ids and for the metadata
    """
    logger.info(f'Importing ids and metadata from {path} for pea_id {pea_id}')
    path = os.path.join(path, pea_id)
    ids_gen = _ids_gen(path)
    metas_gen = _metas_gen(path)
    return ids_gen, metas_gen


def _ids_gen(path: str):
    with open(os.path.join(path, 'ids'), 'r') as ids_fh:
        for l in ids_fh:
            yield l.strip()


def _vecs_gen(path: str):
    with open(os.path.join(path, 'vectors'), 'rb') as vectors_fh:
        while True:
            next_size = vectors_fh.read(BYTE_PADDING)
            next_size = int.from_bytes(next_size, byteorder=sys.byteorder)
            if next_size:
                vec = np.frombuffer(
                    vectors_fh.read(next_size),
                    dtype=DUMP_DTYPE,
                )
                yield vec
            else:
                break


def _metas_gen(path: str):
    with open(os.path.join(path, 'metas'), 'rb') as metas_fh:
        while True:
            next_size = metas_fh.read(BYTE_PADDING)
            next_size = int.from_bytes(next_size, byteorder=sys.byteorder)
            if next_size:
                meta = metas_fh.read(next_size)
                yield meta
            else:
                break


def _get_file_paths(shard_path: str):
    vectors_fp = os.path.join(shard_path, 'vectors')
    metas_fp = os.path.join(shard_path, 'metas')
    ids_fp = os.path.join(shard_path, 'ids')
    return vectors_fp, metas_fp, ids_fp
