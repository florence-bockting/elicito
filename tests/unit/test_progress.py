"""
Unittests for the progress table
"""

from elicito._progress import ProgressTable


def test_progress_table_counts_steps_and_keeps_the_last_fields():
    bar = ProgressTable(
        "Training", total=3, disable=True, loss=float("nan"), best=float("nan")
    )
    best = float("inf")
    for loss in (0.3, 0.1, 0.2):
        best = min(best, loss)
        bar.update(loss=loss, best=best)
    bar.close()

    task = bar._progress.tasks[0]
    assert task.completed == 3
    assert task.fields == {"loss": 0.2, "best": 0.1}
