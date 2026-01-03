from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class RequestMetrics:
    request_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    latency_totals: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    recent_requests: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=30))
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def record(self, route: str, method: str, status_code: int, latency_seconds: float) -> None:
        self.request_counts[route] += 1
        self.latency_totals[route] += latency_seconds
        self.recent_requests.appendleft(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "route": route,
                "method": method,
                "status_code": status_code,
                "latency_ms": round(1000.0 * latency_seconds, 3),
            }
        )

    def route_summary(self) -> dict[str, dict[str, float | int]]:
        return {
            route: {
                "count": count,
                "avg_latency_ms": round(1000.0 * self.latency_totals[route] / max(count, 1), 3),
            }
            for route, count in self.request_counts.items()
        }

    def recent_activity(self) -> list[dict[str, Any]]:
        return list(self.recent_requests)

    def snapshot(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "routes": self.route_summary(),
            "recent_requests": self.recent_activity(),
        }
