"""
Tests of the checks of the targets and the expert answers
"""

import pytest

import elicito as el
from elicito._checks import check_answers, check_target_names


def _answers(**values):
    return el.elicit._expert(values)["data"]


def test_check_answers_accepts_matching_answers():
    targets = [el.quantiles("y0"), el.correlation("cor")]
    check_answers(_answers(y0=[-1.0, 0.0, 1.0], cor=[0.3]), targets)


def test_check_answers_reports_missing_and_unknown():
    targets = [el.quantiles("y0"), el.quantiles("y1")]
    with pytest.raises(ValueError, match=r"Missing answers: \['y1'\].*\['y2'\]"):
        check_answers(_answers(y0=[-1.0, 0.0, 1.0], y2=[0.0, 1.0, 2.0]), targets)


def test_check_answers_rejects_wrong_length():
    targets = [el.quantiles("y0", probs=(0.25, 0.5, 0.75))]
    with pytest.raises(ValueError, match="has 2 values"):
        check_answers(_answers(y0=[0.0, 1.0]), targets)


def test_check_target_names_rejects_duplicates():
    with pytest.raises(ValueError, match="unique"):
        check_target_names([el.quantiles("y0"), el.identity("y0")])


@pytest.mark.parametrize("name", ["prior", "total_loss", "penalty"])
def test_check_target_names_rejects_reserved(name):
    with pytest.raises(ValueError, match="reserved"):
        check_target_names([el.quantiles(name)])
