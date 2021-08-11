import functools
import os
import time
from typing import List

from jina import DocumentArray
from jina.logging.logger import JinaLogger
from jina.enums import LogVerbosity


def _get_non_empty_fields_doc_array(docs: DocumentArray) -> List[str]:
    non_empty_fields = list(docs[0].non_empty_fields)
    for doc in docs[:1]:
        for field in non_empty_fields:
            if field not in doc.non_empty_fields:
                non_empty_fields.pop(field)
    return non_empty_fields


def add_request_logger(logger):
    """
    Add logging functionality to a request function.
    Only shows logs for `JINA_LOG_LEVEL` > info.
    You can set this as an env variable before starting your `Jina` application.

    Example usages:
    >>> from jina import Executor, requests
    >>> my_logger = JinaLogger('MyExecLogger')
    >>>
    >>> class MyExec(Executor):
    >>>     @requests
    >>>     @add_request_logger(my_logger)
    >>>     def index(self, docs, parameters, **kwargs):
    >>>          ...

    :param logger: The logger you want to use
    """
    def decorator(function):
        @functools.wraps(function)
        def wrapper(self, docs, parameters, **kwargs):
            verbose_level = os.environ.get('JINA_LOG_LEVEL', None)
            verbose_level = LogVerbosity.from_string(verbose_level) if verbose_level else None
            if verbose_level is None or verbose_level > LogVerbosity.DEBUG:
                return function(self, docs, parameters, **kwargs)

            if not docs:
                logger.debug('Docs is None. Nothing to monitor')
                return function(self, docs, parameters, **kwargs)

            logger.debug(f'📄 Received request containing {len(docs)} documents.')
            logger.debug(f'📕 Received parameters dictionary: {parameters}')

            if len(docs) > 0:
                non_empty_fields = _get_non_empty_fields_doc_array(docs)
                logger.debug(f'🏷 Non-empty fields {non_empty_fields}')

            start_time = time.time()
            result = function(self, docs, parameters, **kwargs)
            end_time = time.time()
            logger.debug(f'⏱ Elapsed time for request {end_time - start_time} seconds.')
            return result

        return wrapper
    return decorator
