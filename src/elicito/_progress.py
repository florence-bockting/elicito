"""
Shows a progress table for training and initialization
"""

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
    "best": "Best {task.fields[best]:.4g}",
    "skipped": "Skipped {task.fields[skipped]}",
    "lr": "LR {task.fields[lr]:.3g}",
}


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
        self._progress.update(self._task, advance=advance, **fields)

    def close(self) -> None:
        """Stop the display"""
        self._progress.stop()
