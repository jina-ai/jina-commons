# Test suite for the logging.py module
import os

import pytest
from jina import DocumentArray, Document
from jina.logging.logger import JinaLogger

from jina_commons.logging import add_request_logger


class MockLogger:
    def __init__(self):
        self.logs = []

    def info(self, msg: str):
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


@pytest.mark.parametrize(
    ['log_level', 'output_expected'],
    [
        ('debug', True),
        ('info', False),
        ('error', True)
    ]
)
def test_logs_are_streamed_to_std_out_depends_on_level(log_level: str, output_expected: bool, document_array: DocumentArray, capsys):
    logger = JinaLogger('test')
    os.environ['JINA_LOG_LEVEL'] = log_level

    class MyExec:
        @add_request_logger(logger=logger)
        def index(self, docs, parameters, **kwargs):
            pass

    MyExec().index(document_array, parameters={})

    captured = capsys.readouterr()
    assert (len(captured.out) > 0) == output_expected
    os.environ.pop('JINA_LOG_LEVEL')
