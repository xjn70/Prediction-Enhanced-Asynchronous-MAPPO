"""A fixed cyclic service order with independent progress for each AUV."""

from ..entities.common import _require_int
from ..environment.actions import Action, ReturnAction, VisitAction
from ..environment.observation import LocalObservation


class CyclicPatrol:
    def __init__(self, initial_offset: int = 0) -> None:
        _require_int(initial_offset, "initial_offset")
        self._next_index = initial_offset

    def choose_action(self, observation: LocalObservation) -> Action:
        actions = observation.legal_actions
        if actions and isinstance(actions[0], ReturnAction):
            return actions[0]

        eligible = {action.ch_id for action in actions if isinstance(action, VisitAction)}
        order = tuple(ch.ch_id for ch in observation.chs)
        for offset in range(len(order)):
            index = (self._next_index + offset) % len(order)
            ch_id = order[index]
            if ch_id in eligible:
                self._next_index = (index + 1) % len(order)
                return VisitAction(ch_id)
        raise ValueError("CyclicPatrol requires a feasible action")
