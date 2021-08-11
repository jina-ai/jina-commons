# Test suite for the logging.py module
import os

import pytest
from jina import DocumentArray, Document

from jina_commons.logging import add_request_logger


class MockLogger:
    def __init__(self):
        self.logs = []

    def debug(self, msg: str):
        self.logs.append(msg)


@pytest.fixture
def document_array() -> DocumentArray:
    return DocumentArray(
        [
            Document(text='test'),
            Document(text='test'),
            Document(text='test')
        ]
    )


def test_request_logger(document_array: DocumentArray):
    logger = MockLogger()
    os.environ['JINA_LOG_LEVEL'] = 'debug'

    class MyExec:
        @add_request_logger(logger=logger)
        def index(self, docs, parameters, **kwargs):
            pass

    executor = MyExec()

    executor.index(docs=document_array, parameters={'top_k': 3})

    assert len(logger.logs) == 4
    os.environ.pop('JINA_LOG_LEVEL')


def test_request_logger_recognizes_log_level(document_array: DocumentArray):
    logger = MockLogger()

    class MyExec:
        @add_request_logger(logger=logger)
        def index(self, docs, parameters, **kwargs):
            pass

    executor = MyExec()

    executor.index(document_array, parameters={})

    assert len(logger.logs) == 0
