"""Discrete macro actions shared by all baseline policies."""

from dataclasses import dataclass

from ..entities.common import _require_int


@dataclass(frozen=True)
class VisitAction:
    ch_id: int

    def __post_init__(self) -> None:
        _require_int(self.ch_id, "ch_id")


@dataclass(frozen=True)
class ReturnAction:
    """Return to the fixed Sink and upload the cargo."""


Action = VisitAction | ReturnAction
