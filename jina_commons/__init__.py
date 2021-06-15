from jina import Executor

from jina.logging.logger import JinaLogger


def get_logger(self: Executor):
    """get an instance of JinaLogger based on either instance name, or class name"""
    return JinaLogger(getattr(self.metas, 'name', self.__class__.__name__))
