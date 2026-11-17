"""Run the three Task 07 policies on paired scenarios and export CSVs."""

import argparse
import csv
from collections.abc import Iterable
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.default_config import BASELINE_RETURN_RULE
from uwsn.environment import make_scenario
from uwsn.environment.actions import ReturnAction, VisitAction
from uwsn.evaluation import BASELINE_POLICIES, BaselineResult, run_comparison, summarize_metrics
from uwsn.evaluation.metrics import metric_columns


def _write_csv(path: Path, columns: Iterable[str], rows: Iterable[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


DECISION_COLUMNS = (
    "policy", "seed", "auv_id", "decision_time_s", "action", "ch_id", "reason", "report_time_s",
    "arrival_report_time_s", "arrived_at_s", "outcome", "finished_at_s", "collected_packets",
    "collected_bytes", "upload_started_at_s", "upload_finished_at_s", "uploaded_packets",
    "planned_travel_distance_m", "wake_at_s",
)


def _decision_rows(result: BaselineResult) -> Iterable[dict]:
    rows = []
    for visit in result.episode.decisions:
        ch_id = visit.action.ch_id if isinstance(visit.action, VisitAction) else None
        rows.append({
            "policy": result.policy, "seed": result.seed, "auv_id": visit.auv_id,
            "decision_time_s": visit.decision_time_s, "action": "visit" if ch_id is not None else "return",
            "ch_id": ch_id, "reason": visit.reason, "report_time_s": visit.report_time_s,
            "arrival_report_time_s": visit.arrival_report_time_s, "arrived_at_s": visit.arrived_at_s,
            "outcome": visit.outcome, "finished_at_s": visit.finished_at_s,
            "collected_packets": visit.collected_packets, "collected_bytes": visit.collected_bytes,
            "upload_started_at_s": visit.upload_started_at_s, "upload_finished_at_s": visit.upload_finished_at_s,
            "uploaded_packets": visit.uploaded_packets, "planned_travel_distance_m": visit.travel.distance_m,
        })
    for inactive in result.episode.inactive_decisions:
        rejected = inactive.rejected_action
        rows.append({
            "policy": result.policy, "seed": result.seed, "auv_id": inactive.auv_id,
            "decision_time_s": inactive.decision_time_s, "reason": inactive.reason,
            "outcome": "parked" if inactive.wake_at_s is None else "deferred",
            "wake_at_s": inactive.wake_at_s,
            "action": "visit" if isinstance(rejected, VisitAction) else (
                "return" if isinstance(rejected, ReturnAction) else None
            ),
            "ch_id": rejected.ch_id if isinstance(rejected, VisitAction) else None,
        })
    return sorted(rows, key=lambda row: (row["decision_time_s"], row["auv_id"]))


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("S1", "S2", "S3"), default="S1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--horizon-s", type=float, default=900)
    parser.add_argument("--n-auv", type=int, default=3)
    parser.add_argument("--n-ch", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("results/task07"))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain distinct values")

    def scenario(seed: int):
        return make_scenario(
            args.scenario, seed, n_ch=args.n_ch, n_auv=args.n_auv, horizon_s=args.horizon_s,
            record_log=False,
        )

    results = run_comparison(scenario, args.seeds, BASELINE_RETURN_RULE)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenario_fields = {
        "scenario": args.scenario, "horizon_s": args.horizon_s, "n_auv": args.n_auv, "n_ch": args.n_ch,
        "storage_threshold": BASELINE_RETURN_RULE.storage_threshold,
        "return_reserve_s": BASELINE_RETURN_RULE.return_reserve_s,
        "revisit_cooldown_s": BASELINE_RETURN_RULE.revisit_cooldown_s,
    }
    rows = [
        {"policy": result.policy, "seed": result.seed, **scenario_fields, **metric_columns(result.episode.metrics)}
        for result in results
    ]
    _write_csv(args.output_dir / "episodes.csv", rows[0], rows)
    _write_csv(
        args.output_dir / "decisions.csv", DECISION_COLUMNS,
        (row for result in results for row in _decision_rows(result)),
    )

    print(f"Scenario {args.scenario}, {args.n_auv} AUVs, {args.n_ch} CHs, "
          f"{args.horizon_s:g} s, seeds={args.seeds}")
    print(f"{'policy':21} {'seed':>4} {'VoI':>8} {'pkts':>5} {'dropped':>7} "
          f"{'wait(s)':>8} {'busy %':>7} {'travel(m)':>10}  CH arrival counts")
    for result in results:
        metrics = result.episode.metrics
        counts = " ".join(f"{ch_id}:{count}" for ch_id, count in metrics.per_ch_visit_counts)
        print(f"{result.policy:21} {result.seed:4} {metrics.total_delivered_voi:8.2f} "
              f"{metrics.delivered_packets:5} {metrics.dropped_packets:7} "
              f"{_number(metrics.average_sink_wait_s):>8} {metrics.sink_observed_utilization:7.1%} "
              f"{metrics.total_auv_travel_distance_m:10.2f}  {counts}")

    aggregate_rows = []
    print("\nMean +/- sample SD")
    print(f"{'policy':21} {'VoI':>19} {'delivered packets':>19} {'dropped packets':>19} "
          f"{'Sink wait (s)':>19} {'travel (m)':>19}")
    for name, _ in BASELINE_POLICIES:
        summary = summarize_metrics(result.episode.metrics for result in results if result.policy == name)
        display = [
            f"{_number(summary[key].mean)} +/- {_number(summary[key].std)}"
            for key in ("total_delivered_voi", "delivered_packets", "dropped_packets",
                        "average_sink_wait_s", "total_auv_travel_distance_m")
        ]
        print(f"{name:21} " + " ".join(f"{value:>19}" for value in display))
        aggregate_rows.extend({
            "policy": name, "metric": key, "mean": value.mean, "std": value.std,
            "sample_count": value.sample_count,
        } for key, value in summary.items())
    _write_csv(args.output_dir / "summary.csv", ("policy", "metric", "mean", "std", "sample_count"), aggregate_rows)
    print(f"CSV output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
