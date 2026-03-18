"""
run.py

项目入口文件：
1. 读取 YAML 配置
2. 读取 .env 环境变量（可选）
3. 初始化项目目录结构（logs/data/screenshots）
4. 创建 BottomUpAgent 实例
5. 启动智能体主流程
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_TASK = "Play the game"


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bottom-Up-Agent runner")
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/game.yaml",
        help="YAML 配置文件路径，默认使用 config/game.yaml",
    )
    return parser.parse_args()


def load_env_file(project_root: Path) -> None:
    """
    加载根目录下的 .env。
    优先尝试 python-dotenv；若未安装则使用简易解析。
    """
    env_path = project_root / ".env"
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(env_path)
        return
    except Exception:
        pass

    # 简易回退解析，避免没装 python-dotenv 直接报错
    try:
        with env_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)
    except Exception as exc:
        print(f"[WARN] 读取 .env 失败: {exc}")


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"配置文件为空: {path}")
    if not isinstance(config, dict):
        raise TypeError("YAML 顶层结构必须是字典(dict)")

    return config


def normalize_config(config: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """
    补齐默认配置，避免后续模块里出现太多防空判断。
    """
    runtime = config.setdefault("runtime", {})
    runtime.setdefault("task", DEFAULT_TASK)
    runtime.setdefault("max_steps", 10)
    runtime.setdefault("log_level", "INFO")
    runtime.setdefault("mode", "auto")
    runtime.setdefault("stop_on_failures", 3)

    model = config.setdefault("model", {})
    model.setdefault("provider", "mock")
    model.setdefault("name", "demo-model")

    environment = config.setdefault("environment", {})
    environment.setdefault("name", "demo-environment")
    environment.setdefault("window_name", "Demo Window")
    environment.setdefault("resolution", [1280, 720])

    visualization = config.setdefault("visualization", {})
    visualization.setdefault("enabled", True)
    visualization.setdefault("refresh_interval_ms", 300)

    paths = config.setdefault("paths", {})
    paths.setdefault("run_logs", "logs/run_logs")
    paths.setdefault("action_logs", "logs/action_logs")
    paths.setdefault("states", "data/states")
    paths.setdefault("actions", "data/actions")
    paths.setdefault("feedback", "data/feedback")
    paths.setdefault("skills", "data/skills")
    paths.setdefault("screenshots_current", "screenshots/current")
    paths.setdefault("screenshots_history", "screenshots/history")

    config["_project_root"] = str(project_root.resolve())
    return config


def resolve_path(project_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def ensure_project_dirs(config: Dict[str, Any], project_root: Path) -> Dict[str, str]:
    """
    初始化 README 中建议的目录结构。
    """
    raw_paths = config["paths"]
    resolved_paths: Dict[str, str] = {}

    for key, value in raw_paths.items():
        path = resolve_path(project_root, value)
        path.mkdir(parents=True, exist_ok=True)
        resolved_paths[key] = str(path)

    return resolved_paths


def setup_logger(log_level: str, run_log_file: Path) -> None:
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(run_log_file, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def inject_runtime_context(
    config: Dict[str, Any],
    project_root: Path,
    resolved_paths: Dict[str, str],
    config_file: str,
) -> Dict[str, Any]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]

    runtime_context = {
        "run_id": run_id,
        "project_root": str(project_root.resolve()),
        "config_file": str((project_root / config_file).resolve())
        if not Path(config_file).is_absolute()
        else str(Path(config_file).resolve()),
        "start_time": now_str(),
        "paths": resolved_paths,
    }

    config["_runtime_context"] = runtime_context
    return runtime_context


def save_bootstrap_summary(config: Dict[str, Any], runtime_context: Dict[str, Any]) -> None:
    """
    保存一次启动快照，便于演示和调试。
    """
    run_logs_dir = Path(runtime_context["paths"]["run_logs"])
    bootstrap_file = run_logs_dir / f"bootstrap_{runtime_context['run_id']}.json"

    payload = {
        "run_id": runtime_context["run_id"],
        "start_time": runtime_context["start_time"],
        "config_file": runtime_context["config_file"],
        "task": config["runtime"]["task"],
        "max_steps": config["runtime"]["max_steps"],
        "mode": config["runtime"]["mode"],
        "model": config["model"],
        "environment": config["environment"],
        "paths": runtime_context["paths"],
    }

    with bootstrap_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_agent(config: Dict[str, Any]):
    from BottomUpAgent.common.BottomUpAgent import BottomUpAgent
    return BottomUpAgent(config=config)


def main() -> int:
    try:
        args = parse_args()
        project_root = Path(__file__).resolve().parent

        load_env_file(project_root)

        raw_config = load_yaml_config(args.config_file)
        config = normalize_config(raw_config, project_root)
        resolved_paths = ensure_project_dirs(config, project_root)
        runtime_context = inject_runtime_context(
            config=config,
            project_root=project_root,
            resolved_paths=resolved_paths,
            config_file=args.config_file,
        )

        run_log_file = Path(resolved_paths["run_logs"]) / f"run_{runtime_context['run_id']}.log"
        setup_logger(config["runtime"]["log_level"], run_log_file)

        logging.info("配置文件加载成功: %s", runtime_context["config_file"])
        logging.info("运行 ID: %s", runtime_context["run_id"])
        logging.info("任务: %s", config["runtime"]["task"])
        logging.info("模型: %s / %s", config["model"]["provider"], config["model"]["name"])
        logging.info("环境: %s", config["environment"]["name"])
        logging.info("最大步数: %s", config["runtime"]["max_steps"])

        save_bootstrap_summary(config, runtime_context)

        agent = build_agent(config)
        startup_delay = float(config.get("runtime", {}).get("startup_delay_seconds", 0))
        if startup_delay > 0:
            logging.info("启动延迟 %.1f 秒，请切到游戏窗口。", startup_delay)
            time.sleep(startup_delay)
        result = agent.run(task=config["runtime"]["task"])

        summary_file = Path(resolved_paths["run_logs"]) / f"run_summary_{runtime_context['run_id']}.json"
        with summary_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logging.info("运行完成，结果摘要已保存: %s", summary_file)
        return 0

    except Exception as exc:
        logging.exception("程序运行失败: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())