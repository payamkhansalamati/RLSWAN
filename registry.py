"""Controller registry and validation helpers for the final 4-controller system."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

# Final system controllers (authoritative order used across train/evaluate).
ALL_CONTROLLERS: Tuple[str, ...] = (
    "pid",
    "vanilla_ppo",
    "context_ppo",
    "hybrid_context_pid",
)

# Trainable PPO policies in this project.
TRAINABLE_CONTROLLERS: Tuple[str, ...] = (
    "vanilla_ppo",
    "context_ppo",
)

# PPO environment wiring for train/load code paths.
PPO_AGENT_SPECS: Dict[str, Tuple[bool, bool]] = {
    "vanilla_ppo": (False, False),
    "context_ppo": (True, True),
}


def validate_controllers(
    controllers: Iterable[str],
    *,
    allow: Iterable[str],
    context: str,
) -> List[str]:
    """Validate controller names and return a normalized list.

    Raises:
        ValueError: if any unknown controller is provided.
    """
    allowed = tuple(allow)
    invalid = [name for name in controllers if name not in allowed]
    if invalid:
        allowed_str = ", ".join(allowed)
        invalid_str = ", ".join(sorted(set(invalid)))
        raise ValueError(
            f"Unknown controller(s) for {context}: {invalid_str}. "
            f"Allowed: {allowed_str}."
        )
    return list(controllers)


def get_ppo_spec(agent_type: str) -> Tuple[bool, bool]:
    """Return (use_context, use_context_reward) for a trainable PPO controller."""
    if agent_type not in PPO_AGENT_SPECS:
        allowed = ", ".join(TRAINABLE_CONTROLLERS)
        raise ValueError(
            f"Unknown trainable controller: {agent_type}. Allowed: {allowed}."
        )
    return PPO_AGENT_SPECS[agent_type]
