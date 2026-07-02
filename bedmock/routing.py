"""Route Bedrock model IDs to provider profiles and target models."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass

from .config import BedmockConfig, Route
from .exceptions import ConfigurationError


@dataclass
class RouteResolution:
    provider_id: str
    target_model: str
    source_codec: str | None
    route_id: str


def _match_route(route: Route, model_id: str) -> int | None:
    match = route.match
    if match.get("model_id") == model_id:
        return 1
    glob = match.get("model_id_glob")
    if glob and fnmatch.fnmatchcase(model_id, str(glob)):
        return 2
    regex = match.get("model_id_regex")
    if regex and re.search(str(regex), model_id):
        return 3
    return None


def resolve_route(config: BedmockConfig, model_id: str) -> RouteResolution:
    matches: list[tuple[int, Route]] = []
    for route in config.routes:
        priority = _match_route(route, model_id)
        if priority is not None:
            matches.append((priority, route))

    if matches:
        matches.sort(key=lambda item: item[0])
        best_priority = matches[0][0]
        best = [route for priority, route in matches if priority == best_priority]
        if len(best) > 1:
            ids = ", ".join(route.id for route in best)
            raise ConfigurationError(f"Multiple Bedmock routes match {model_id!r}: {ids}")
        route = best[0]
        return RouteResolution(
            provider_id=route.target.provider,
            target_model=route.target.model,
            source_codec=route.source_codec,
            route_id=route.id,
        )

    if config.provider and config.model:
        return RouteResolution(
            provider_id=config.provider,
            target_model=config.model,
            source_codec=None,
            route_id="environment",
        )

    raise ConfigurationError(
        "No Bedmock route resolved. Set BEDMOCK_PROVIDER and BEDMOCK_MODEL or create bedmock.json."
    )
