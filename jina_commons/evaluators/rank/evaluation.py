from typing import Sequence, Any, Optional

from typing import Sequence, Union, Optional, Tuple, Any
from math import log

from jina.executors.evaluators.rank import BaseRankingEvaluator


def _compute_dcg(gains, power_relevance):
    """Compute discounted cumulative gain."""
    ret = 0.0
    if not power_relevance:
        for score, position in zip(gains[1:], range(2, len(gains) + 1)):
            ret += score / log(position, 2)
        return gains[0] + ret
    for score, position in zip(gains, range(1, len(gains) + 1)):
        ret += (pow(2, score) - 1) / log(position + 1, 2)
    return ret


def _compute_idcg(gains, power_relevance):
    """Compute ideal discounted cumulative gain."""
    sorted_gains = sorted(gains, reverse=True)
    return _compute_dcg(sorted_gains, power_relevance)


class NDCGEvaluator(BaseRankingEvaluator):
    """
    From a sorted list of retrieved, scores and scores,
    evaluates normalized discounted cumulative gain for information retrieval.
    :param eval_at: The number of documents in each of the lists to consider
        in the NDCG computation. If ``None``is given, the complete lists are
        considered for the evaluation.
    :param power_relevance: The power relevance places stronger emphasis on
        retrieving relevant documents. For detailed information, please check
        https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    :param is_relevance_score: Boolean indicating if the actual scores are
        to be considered relevance. Highest value is better.
        If True, the information coming from the actual system results will
        be sorted in descending order, otherwise in ascending order.
        Since the input of the evaluate method is sorted according to the
        `scores` of both actual and desired input, this parameter is
        useful for instance when the ``matches` come directly from a ``VectorIndexer``
        where score is `distance` and therefore the smaller the better.
    .. note:
        All the IDs that are not found in the ground truth will be considered to have
        relevance 0.
    """

    def __init__(self,
                 eval_at: Optional[int] = None,
                 power_relevance: bool = True,
                 is_relevance_score: bool = True):
        super().__init__()
        self._eval_at = eval_at
        self._power_relevance = power_relevance
        self._is_relevance_score = is_relevance_score

    def evaluate(
            eval_at: Optional[int],
            actual: Sequence[Tuple[Any, Union[int, float]]],
            desired: Sequence[Tuple[Any, Union[int, float]]],
            power_relevance: bool = True,
            is_relevance_score: bool = True,
            *args, **kwargs
    ) -> float:
        """"
        Evaluate normalized discounted cumulative gain for information retrieval.
        :param eval_at: The number of documents in each of the lists to consider
            in the NDCG computation. If ``None``is given, the complete lists are
            considered for the evaluation.
        :param actual: The tuple of Ids and Scores predicted by the search system.
            They will be sorted in descending order.
        :param desired: The expected id and relevance tuples given by user as
            matching round truth.
        :param power_relevance: The power relevance places stronger emphasis on
            retrieving relevant documents. For detailed information, please check
        https://en.wikipedia.org/wiki/Discounted_cumulative_gain
        :param is_relevance_score: Boolean indicating if the actual scores are
            to be considered relevance. Highest value is better.
            If True, the information coming from the actual system results will
            be sorted in descending order, otherwise in ascending order.
            Since the input of the evaluate method is sorted according to the
            `scores` of both actual and desired input, this parameter is
            useful for instance when the ``matches` come directly from a ``VectorIndexer``
            where score is `distance` and therefore the smaller the better.
        :param args:  Additional positional arguments
        :param kwargs: Additional keyword arguments
        :return: The evaluation metric value for the request document.
        """

        def _compute_dcg(gains, power_relevance):
            """Compute discounted cumulative gain."""
            ret = 0.0
            if not power_relevance:
                for score, position in zip(gains[1:], range(2, len(gains) + 1)):
                    ret += score / log(position, 2)
                return gains[0] + ret
            for score, position in zip(gains, range(1, len(gains) + 1)):
                ret += (pow(2, score) - 1) / log(position + 1, 2)
            return ret

        def _compute_idcg(gains, power_relevance):
            """Compute ideal discounted cumulative gain."""
            sorted_gains = sorted(gains, reverse=True)
            return _compute_dcg(sorted_gains, power_relevance)

        relevances = dict(desired)
        actual_relevances = list(map(lambda x: relevances[x[0]] if x[0] in relevances else 0.,
                                     sorted(actual, key=lambda x: x[1], reverse=is_relevance_score)))
        desired_relevances = list(map(lambda x: x[1], sorted(desired, key=lambda x: x[1], reverse=True)))

        # Information gain must be greater or equal to 0.
        actual_at_k = actual_relevances[:eval_at] if eval_at else actual
        desired_at_k = desired_relevances[:eval_at] if eval_at else desired
        if not actual_at_k:
            raise ValueError(f'Expecting gains at k with minimal length of 1, {len(actual_at_k)} received.')
        if not desired_at_k:
            raise ValueError(f'Expecting desired at k with minimal length of 1, {len(desired_at_k)} received.')
        if any(item < 0 for item in actual_at_k) or any(item < 0 for item in desired_at_k):
            raise ValueError('One or multiple score is less than 0.')
        dcg = _compute_dcg(gains=actual_at_k, power_relevance=power_relevance)
        idcg = _compute_idcg(gains=desired_at_k, power_relevance=power_relevance)
        return 0.0 if idcg == 0.0 else dcg / idcg


def precision(
        eval_at: Optional[int], actual: Sequence[Any], desired: Sequence[Any], *args, **kwargs
) -> float:
    """
    Compute precision evaluation score

    :param eval_at: the point at which evaluation is computed, if None give, will consider all the input to evaluate
    :param actual: the matched document identifiers from the request as matched by jina indexers and rankers
    :param desired: the expected documents matches ids sorted as they are expected
    :return the evaluation metric value for the request document
    """
    if eval_at == 0:
        return 0.0
    actual_at_k = actual[: eval_at] if eval_at else actual
    ret = len(set(actual_at_k).intersection(set(desired)))
    sub = len(actual_at_k)
    return ret / sub if sub != 0 else 0.0


def recall(
        eval_at: Optional[int], actual: Sequence[Any], desired: Sequence[Any], *args, **kwargs
) -> float:
    """
    Compute precision evaluation score

    :param eval_at: the point at which evaluation is computed, if None give, will consider all the input to evaluate
    :param actual: the matched document identifiers from the request as matched by jina indexers and rankers
    :param desired: the expected documents matches ids sorted as they are expected
    :return the evaluation metric value for the request document
    """
    if eval_at == 0:
        return 0.0
    actual_at_k = actual[: eval_at] if eval_at else actual
    ret = len(set(actual_at_k).intersection(set(desired)))
    return ret / len(desired)


def ndcg(
        eval_at: Optional[int],
        actual: Sequence[Tuple[Any, Union[int, float]]],
        desired: Sequence[Tuple[Any, Union[int, float]]],
        power_relevance: bool = True,
        is_relevance_score: bool = True,
        *args, **kwargs
) -> float:
    """"
    Evaluate normalized discounted cumulative gain for information retrieval.
    :param eval_at: The number of documents in each of the lists to consider
        in the NDCG computation. If ``None``is given, the complete lists are
        considered for the evaluation.
    :param actual: The tuple of Ids and Scores predicted by the search system.
        They will be sorted in descending order.
    :param desired: The expected id and relevance tuples given by user as
        matching round truth.
    :param power_relevance: The power relevance places stronger emphasis on
        retrieving relevant documents. For detailed information, please check
    https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    :param is_relevance_score: Boolean indicating if the actual scores are
        to be considered relevance. Highest value is better.
        If True, the information coming from the actual system results will
        be sorted in descending order, otherwise in ascending order.
        Since the input of the evaluate method is sorted according to the
        `scores` of both actual and desired input, this parameter is
        useful for instance when the ``matches` come directly from a ``VectorIndexer``
        where score is `distance` and therefore the smaller the better.
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    :return: The evaluation metric value for the request document.
    """

    def _compute_dcg(gains):
        """Compute discounted cumulative gain."""
        ret = 0.0
        if not power_relevance:
            for score, position in zip(gains[1:], range(2, len(gains) + 1)):
                ret += score / log(position, 2)
            return gains[0] + ret
        for score, position in zip(gains, range(1, len(gains) + 1)):
            ret += (pow(2, score) - 1) / log(position + 1, 2)
        return ret

    def _compute_idcg(gains):
        """Compute ideal discounted cumulative gain."""
        sorted_gains = sorted(gains, reverse=True)
        return _compute_dcg(sorted_gains)

    relevances = dict(desired)
    actual_relevances = list(map(lambda x: relevances[x[0]] if x[0] in relevances else 0.,
                                 sorted(actual, key=lambda x: x[1], reverse=is_relevance_score)))
    desired_relevances = list(map(lambda x: x[1], sorted(desired, key=lambda x: x[1], reverse=True)))

    # Information gain must be greater or equal to 0.
    actual_at_k = actual_relevances[:eval_at] if eval_at else actual
    desired_at_k = desired_relevances[:eval_at] if eval_at else desired
    if not actual_at_k:
        raise ValueError(f'Expecting gains at k with minimal length of 1, {len(actual_at_k)} received.')
    if not desired_at_k:
        raise ValueError(f'Expecting desired at k with minimal length of 1, {len(desired_at_k)} received.')
    if any(item < 0 for item in actual_at_k) or any(item < 0 for item in desired_at_k):
        raise ValueError('One or multiple score is less than 0.')
    dcg = _compute_dcg(gains=actual_at_k, power_relevance=power_relevance)
    idcg = _compute_idcg(gains=desired_at_k, power_relevance=power_relevance)
    return 0.0 if idcg == 0.0 else dcg / idcg


def reciprocal_rank(
        actual: Sequence[Any], desired: Sequence[Any], *args, **kwargs
) -> float:
    """
    Evaluate score as per reciprocal rank metric.
    :param actual: Sequence of sorted document IDs.
    :param desired: Sequence of sorted relevant document IDs
        (the first is the most relevant) and the one to be considered.
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    :return: Reciprocal rank score
    """
    if len(actual) == 0 or len(desired) == 0:
        return 0.0
    try:
        return 1.0 / (actual.index(desired[0]) + 1)
    except:
        return 0.0


def average_precision(actual: Sequence[Any], desired: Sequence[Any], *args, **kwargs) -> float:
    """"
    Evaluate the Average Precision of the search.
    :param actual: the matched document identifiers from the request
        as matched by Indexers and Rankers
    :param desired: A list of all the relevant IDs. All documents
        identified in this list are considered to be relevant
    :return: the evaluation metric value for the request document
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    if len(desired) == 0 or len(actual) == 0:
        return 0.0

    precisions = list(map(lambda eval_at: precisions(eval_at, actual, desired), range(1, len(actual) + 1)))
    return sum(precisions) / len(desired)


def fscore(eval_at: Optional[int], actual: Sequence[Any], desired: Sequence[Any], beta: int = 1, *args, **kwargs) -> float:
    """"
    Evaluate the f-score of the search.
    :param eval_at: The point at which precision and recall are computed,
        if ``None`` is given, all input will be considered to evaluate.
    :param actual: The matched document identifiers from the request
        as matched by jina indexers and rankers
    :param desired: The expected documents matches
    :param beta: Parameter to weight differently precision and recall.
        When ``beta` is 1, the fScore corresponds to the harmonic mean
        of precision and recall
    :param args: Additional positional arguments
    :param kwargs: Additional keyword arguments
    :return: the evaluation metric value for the request document
    """
    assert beta != 0, 'fScore is not defined for beta 0'
    weight = beta ** 2
    if not desired or self.eval_at == 0:
        return 0.0

    actual_at_k = actual[:eval_at] if eval_at else actual
    common_count = len(set(actual_at_k).intersection(set(desired)))
    recall = common_count / len(desired)

    divisor = min(eval_at or len(desired), len(desired))

    if divisor != 0.0:
        precision = common_count / divisor
    else:
        precision = 0

    if precision + recall == 0:
        return 0

    return ((1 + weight) * precision * recall) / ((weight * precision) + recall)
