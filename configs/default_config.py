"""Shared Task 07 experiment configuration."""

from uwsn.environment.action_rules import ReturnRuleConfig


BASELINE_RETURN_RULE = ReturnRuleConfig(
    storage_threshold=0.75,
    return_reserve_s=30.0,
    revisit_cooldown_s=20.0,
)
