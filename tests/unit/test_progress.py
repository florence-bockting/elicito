"""
Unittests for the progress table
"""

from queue import Queue

from elicito._progress import _WORKER, ProgressTable, SeedTable, run_in_worker


def test_progress_table_counts_steps_and_keeps_the_last_fields():
    bar = ProgressTable("Training", total=3, disable=True, loss=float("nan"), skipped=0)
    for skipped, loss in enumerate((0.3, 0.1, 0.2)):
        bar.update(loss=loss, skipped=skipped)
    bar.close()

    task = bar._progress.tasks[0]
    assert task.completed == 3
    assert task.fields == {"loss": 0.2, "skipped": 2}


def test_seed_table_shows_the_worker_table_in_the_row_of_its_seed():
    queue: Queue = Queue()
    table = SeedTable([11, 22], queue, disable=True)

    def train(seed):
        bar = ProgressTable("Training", total=3, disable=False, loss=float("nan"))
        for loss in (0.3, 0.1, 0.2):
            bar.update(loss=loss)
        bar.close()
        return seed

    assert run_in_worker(train, 22, 1, queue) == 22
    table.close()

    waiting, trained = table._progress.tasks
    assert waiting.description == "Waiting"
    assert trained.description == "Training"
    assert trained.completed == 3
    assert trained.fields == {"seed": 22, "info": "Loss 0.2"}
    assert _WORKER["queue"] is None
