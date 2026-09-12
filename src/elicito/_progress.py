"""
Shows a progress table for training and initialization
"""

import threading
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

# the text of each optional column
FIELD_FORMATS = {
    "loss": "Loss {task.fields[loss]:.4g}",
    "skipped": "Skipped {task.fields[skipped]}",
}

# a worker of a parallel fit sets the queue and its row. Its tables then send
# their updates to the parent process, and draw nothing.
_WORKER: dict[str, Any] = {"queue": None, "row": None}


def _format_fields(fields: dict[str, Any]) -> str:
    """Write the named fields as one text, in the format of `FIELD_FORMATS`"""
    task = SimpleNamespace(fields=fields)
    return " ".join(FIELD_FORMATS[name].format(task=task) for name in fields)


class SpeedColumn(ProgressColumn):
    """
    Shows the number of steps in each second
    """

    def render(self, task: Task) -> Text:
        """Render the speed of the task"""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("- it/s")
        return Text(f"{speed:.1f} it/s")


class ProgressTable:
    """
    Shows one row with a bar, the named fields, the speed and the time

    In a worker of a parallel fit, the row goes to the parent process.

    Parameters
    ----------
    description
        the label at the start of the row
    total
        the number of steps
    disable
        if True, nothing is shown
    **fields
        the start value of each column in `FIELD_FORMATS` to show
    """

    def __init__(
        self, description: str, total: int, disable: bool, **fields: Any
    ) -> None:
        self._queue = None if disable else _WORKER["queue"]
        if self._queue is not None:
            self._row = _WORKER["row"]
            self._fields = fields
            self._queue.put(
                ("start", self._row, description, total, _format_fields(fields))
            )
            return

        self._progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            *(TextColumn(FIELD_FORMATS[name]) for name in fields),
            SpeedColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            disable=disable,
        )
        self._task = self._progress.add_task(description, total=total, **fields)
        self._progress.start()

    def update(self, advance: int = 1, **fields: Any) -> None:
        """Advance the bar and set new field values"""
        if self._queue is not None:
            self._fields.update(fields)
            self._queue.put(
                ("update", self._row, advance, _format_fields(self._fields))
            )
            return
        self._progress.update(self._task, advance=advance, **fields)

    def close(self) -> None:
        """Stop the display"""
        if self._queue is not None:
            return
        # rich skips the last redraw in Jupyter, so the table would keep the
        # state of the last timed redraw
        self._progress.refresh()
        self._progress.stop()


def run_in_worker(
    function: Callable[[int], Any], seed: int, row: int, queue: Any
) -> Any:
    """
    Run `function(seed)` in a worker, and send its tables to `queue`

    Parameters
    ----------
    function
        the function to run
    seed
        the seed given to the function
    row
        the row of this worker in the `SeedTable`
    queue
        the queue that the `SeedTable` of the parent process reads

    Returns
    -------
    :
        the return value of the function
    """
    _WORKER.update(queue=queue, row=row)
    try:
        return function(seed)
    finally:
        _WORKER.update(queue=None, row=None)


class SeedTable:
    """
    Shows one row for each seed of a parallel fit

    The workers send their updates through `queue`. A thread reads the queue
    and draws the rows.

    Parameters
    ----------
    seeds
        the seed of each row
    queue
        the queue that the workers write to
    disable
        if True, nothing is shown
    """

    def __init__(self, seeds: Sequence[int], queue: Any, disable: bool) -> None:
        self._queue = queue
        self._progress = Progress(
            TextColumn("Seed {task.fields[seed]}"),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[info]}"),
            SpeedColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            disable=disable,
        )
        self._tasks = [
            self._progress.add_task("Waiting", total=None, seed=seed, info="")
            for seed in seeds
        ]
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._progress.start()
        self._reader.start()

    def _read(self) -> None:
        """Apply the updates from the queue until it sends None"""
        while (message := self._queue.get()) is not None:
            kind, row, *values = message
            if kind == "start":
                description, total, info = values
                # reset replaces all fields if it gets any, so the fields go
                # to update, which keeps the seed
                self._progress.reset(
                    self._tasks[row], total=total, description=description
                )
                self._progress.update(self._tasks[row], info=info)
            else:
                advance, info = values
                self._progress.update(self._tasks[row], advance=advance, info=info)

    def close(self) -> None:
        """Read the last updates, and stop the display"""
        self._queue.put(None)
        self._reader.join()
        # rich skips the last redraw in Jupyter
        self._progress.refresh()
        self._progress.stop()
