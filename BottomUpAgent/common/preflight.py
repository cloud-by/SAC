
from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PreflightReport:
    ok: bool = True
    checks: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def add(self, name: str, status: str, message: str, **extra: Any) -> None:
        payload = {"name": name, "status": status, "message": message}
        if extra:
            payload.update(extra)
        self.checks.append(payload)
        if status == "warning":
            self.warnings.append(message)
        elif status == "error":
            self.ok = False
            self.errors.append(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": self.checks,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def run_preflight_checks(config: Dict[str, Any]) -> PreflightReport:
    runtime = dict(config.get("runtime", {}) or {})
    environment = dict(config.get("environment", {}) or {})
    model = dict(config.get("model", {}) or {})

    dry_run = bool(runtime.get("dry_run", True))
    capture_mode = str(environment.get("capture_mode", "screen") or "screen").lower()
    window_name = str(environment.get("window_name", "") or "").strip()
    provider = str(model.get("provider", "mock") or "mock").lower()

    report = PreflightReport()
    system_name = platform.system().lower()
    is_windows = system_name == "windows"

    report.add(
        "runtime_mode",
        "pass",
        f"启动模式检查通过，dry_run={dry_run}, capture_mode={capture_mode}, provider={provider}",
        dry_run=dry_run,
        capture_mode=capture_mode,
        provider=provider,
    )

    _check_window_capture(report, capture_mode=capture_mode, window_name=window_name, dry_run=dry_run, is_windows=is_windows)
    _check_screenshot_backend(report, capture_mode=capture_mode, dry_run=dry_run, is_windows=is_windows)
    _check_input_backend(report, dry_run=dry_run)
    _check_model_provider(report, provider=provider, dry_run=dry_run)

    return report


def _check_window_capture(
    report: PreflightReport,
    *,
    capture_mode: str,
    window_name: str,
    dry_run: bool,
    is_windows: bool,
) -> None:
    if capture_mode != "window":
        report.add("window_target", "pass", f"未启用 window 捕获模式，当前模式为 {capture_mode}。")
        return

    if not window_name:
        report.add("window_target", "error", "capture_mode=window 但未配置 environment.window_name。")
        return

    if not is_windows:
        status = "warning" if dry_run else "error"
        report.add(
            "window_target",
            status,
            f"window 捕获当前主要面向 Windows；当前系统为 {platform.system()}，窗口探测稳定性不足。",
            window_name=window_name,
        )
        return

    try:
        import pygetwindow as gw  # type: ignore

        titles = []
        for item in gw.getAllWindows():
            title = (getattr(item, "title", "") or "").strip()
            if title:
                titles.append(title)

        normalized_target = " ".join(window_name.split()).strip().lower()
        matched = any(
            normalized_target in " ".join(title.split()).strip().lower()
            for title in titles
        )
        if matched:
            report.add("window_target", "pass", f"已发现匹配窗口标题: {window_name}")
        else:
            status = "warning" if dry_run else "error"
            report.add(
                "window_target",
                status,
                f"未发现匹配窗口标题: {window_name}",
                visible_window_count=len(titles),
            )
    except Exception as exc:
        status = "warning" if dry_run else "error"
        report.add("window_target", status, f"窗口探测失败: {exc}")


def _check_screenshot_backend(
    report: PreflightReport,
    *,
    capture_mode: str,
    dry_run: bool,
    is_windows: bool,
) -> None:
    del capture_mode
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    imagegrab_ok = False
    imagegrab_error = ""
    try:
        from PIL import ImageGrab  # type: ignore

        imagegrab_ok = hasattr(ImageGrab, "grab")
    except Exception as exc:
        imagegrab_error = str(exc)

    mss_ok = False
    mss_error = ""
    try:
        import mss  # type: ignore

        mss_ok = hasattr(mss, "mss")
    except Exception as exc:
        mss_error = str(exc)

    if is_windows and (imagegrab_ok or mss_ok):
        report.add("screenshot_backend", "pass", "已发现可用截图后端。", imagegrab=imagegrab_ok, mss=mss_ok)
        return

    if has_display and (imagegrab_ok or mss_ok):
        report.add(
            "screenshot_backend",
            "pass",
            "已发现可用截图后端，且当前图形显示环境存在。",
            imagegrab=imagegrab_ok,
            mss=mss_ok,
        )
        return

    status = "warning" if dry_run else "error"
    detail = "未检测到可用图形显示环境"
    if imagegrab_error or mss_error:
        detail = f"{detail}；ImageGrab={imagegrab_error or 'ok'}；mss={mss_error or 'ok'}"
    report.add("screenshot_backend", status, detail, imagegrab=imagegrab_ok, mss=mss_ok, has_display=has_display)


def _check_input_backend(report: PreflightReport, *, dry_run: bool) -> None:
    if dry_run:
        report.add("input_backend", "pass", "当前 dry_run=True，不强制要求 pyautogui 可用。")
        return
    try:
        dpi_detail = "unknown"
        if platform.system().lower() == "windows":
            try:
                import ctypes

                user32 = ctypes.windll.user32
                shcore = getattr(ctypes.windll, "shcore", None)
                if shcore is not None:
                    try:
                        shcore.SetProcessDpiAwareness(2)
                        dpi_detail = "per_monitor_v2"
                    except Exception:
                        pass
                if dpi_detail == "unknown":
                    try:
                        user32.SetProcessDPIAware()
                        dpi_detail = "system_aware"
                    except Exception:
                        dpi_detail = "set_failed"
            except Exception:
                dpi_detail = "init_failed"
        import pyautogui  # type: ignore

        report.add(
            "input_backend",
            "pass",
            f"pyautogui 已就绪，FAILSAFE={getattr(pyautogui, 'FAILSAFE', None)}",
            dpi_awareness=dpi_detail,
        )
    except Exception as exc:
        report.add("input_backend", "error", f"真实执行模式需要 pyautogui，但加载失败: {exc}")


def _check_model_provider(report: PreflightReport, *, provider: str, dry_run: bool) -> None:
    if provider == "mock":
        status = "warning" if not dry_run else "pass"
        message = "当前使用 mock provider，更适合开发/演示，不适合真实自动化决策。"
        report.add("model_provider", status, message)
        return
    report.add("model_provider", "pass", f"当前模型 provider={provider}")