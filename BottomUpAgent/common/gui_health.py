from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class GuiHealthReport:
    ok: bool = True
    successful_captures: int = 0
    failed_captures: int = 0
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "successful_captures": self.successful_captures,
            "failed_captures": self.failed_captures,
            "details": self.details,
        }


def run_gui_health_check(observer: Any, *, capture_count: int = 1, phase: str = "healthcheck") -> GuiHealthReport:
    report = GuiHealthReport()

    for index in range(1, max(1, int(capture_count)) + 1):
        try:
            state = observer.observe(step_id=index, phase=phase)
            if not isinstance(state, dict):
                raise TypeError("observe() 未返回 dict")

            scene_type = str(state.get("scene_type", "unknown") or "unknown")
            screen_image = str(state.get("screen_image", "") or "")
            report.successful_captures += 1
            report.details.append(
                {
                    "capture_index": index,
                    "status": "pass",
                    "scene_type": scene_type,
                    "screen_image": screen_image,
                }
            )
        except Exception as exc:
            report.ok = False
            report.failed_captures += 1
            report.details.append(
                {
                    "capture_index": index,
                    "status": "error",
                    "message": str(exc),
                }
            )

    if report.failed_captures > 0:
        report.ok = False
    return report