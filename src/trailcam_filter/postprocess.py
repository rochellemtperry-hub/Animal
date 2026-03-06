from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RunSummary:
    processed: int = 0
    animals: int = 0
    non_animals: int = 0

    def on_result(self, has_animal: bool) -> None:
        self.processed += 1
        if has_animal:
            self.animals += 1
        else:
            self.non_animals += 1
