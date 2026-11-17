"""Visit the nearest feasible three-dimensional communication sphere."""

from collections.abc import Iterable

from ..environment.actions import Action, ReturnAction, VisitAction
from ..environment.observation import CHObservation, LocalObservation


def feasible_sites(observation: LocalObservation) -> list[CHObservation]:
    eligible = {action.ch_id for action in observation.legal_actions if isinstance(action, VisitAction)}
    return [ch for ch in observation.chs if ch.ch_id in eligible]


def nearest_visit(observation: LocalObservation, sites: Iterable[CHObservation]) -> VisitAction:
    closest = min(sites, key=lambda ch: (observation.visit_leg(ch).distance_m, ch.ch_id))
    return VisitAction(closest.ch_id)


class NearestNeighbor:
    def choose_action(self, observation: LocalObservation) -> Action:
        actions = observation.legal_actions
        if actions and isinstance(actions[0], ReturnAction):
            return actions[0]
        return nearest_visit(observation, feasible_sites(observation))
