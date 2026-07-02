"""Capability helpers."""

from __future__ import annotations

import fnmatch
import re
from typing import Any, cast

CapabilityValue = bool | str


def resolve_capability(
    provider_capabilities: dict[str, CapabilityValue],
    capability: str,
    *,
    model: str,
    model_overrides: list[dict[str, Any]] | None = None,
) -> CapabilityValue:
    model_overrides = model_overrides or []
    for rule in model_overrides:
        exact = rule.get("model")
        glob = rule.get("model_glob")
        regex = rule.get("model_regex")
        if (
            exact == model
            or (glob and fnmatch.fnmatchcase(model, glob))
            or (regex and re.search(regex, model))
        ):
            caps = rule.get("capabilities", {})
            if capability in caps:
                return cast(CapabilityValue, caps[capability])
    return provider_capabilities.get(capability, "unknown")
