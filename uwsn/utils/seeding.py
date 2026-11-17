"""Deterministic seed derivation for scenario components."""

from hashlib import sha256

from ..entities.common import _require_int


def derive_seed(master_seed: int, role: str, index: int = 0) -> int:
    """Return a stable 64-bit seed for a role and index under a master seed."""
    _require_int(master_seed, "master_seed")
    _require_int(index, "index")
    if not isinstance(role, str) or not role:
        raise ValueError("role must be a nonempty string")
    digest = sha256(f"{master_seed}|{role}|{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
