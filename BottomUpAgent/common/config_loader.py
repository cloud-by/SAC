from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple


def load_yaml_file(path: str | Path) -> Any:
    """加载 YAML 文件；优先使用 PyYAML，缺失时回退到内置简易解析器。"""
    content = Path(path).read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore

        return yaml.safe_load(content)
    except Exception:
        return _simple_yaml_load(content)


def _simple_yaml_load(content: str) -> Any:
    """仅覆盖当前项目配置所需能力的 YAML 子集解析器。"""
    lines: List[Tuple[int, str]] = []
    for raw in content.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw) - len(raw.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(f"仅支持 2 空格缩进，非法行: {raw}")
        lines.append((indent, stripped))

    if not lines:
        return None

    index = 0

    def parse_block(expected_indent: int) -> Any:
        nonlocal index
        if index >= len(lines):
            return None

        _, current = lines[index]
        if current.startswith("-"):
            return parse_list(expected_indent)
        return parse_dict(expected_indent)

    def parse_list(expected_indent: int) -> List[Any]:
        nonlocal index
        result: List[Any] = []

        while index < len(lines):
            indent, text = lines[index]
            if indent < expected_indent:
                break
            if indent != expected_indent or not text.startswith("-"):
                break

            payload = text[1:].strip()
            index += 1

            if not payload:
                result.append(parse_block(expected_indent + 2))
                continue

            if ":" in payload:
                key, value_str = payload.split(":", 1)
                key = key.strip()
                value_str = value_str.strip()
                item: dict[str, Any] = {key: _parse_scalar(value_str) if value_str else None}
                if not value_str:
                    item[key] = parse_block(expected_indent + 2)

                while index < len(lines):
                    child_indent, child_text = lines[index]
                    if child_indent < expected_indent + 2:
                        break
                    if child_indent != expected_indent + 2:
                        raise ValueError(f"非法缩进: {child_text}")
                    if child_text.startswith("-"):
                        break

                    child_key, child_value = child_text.split(":", 1)
                    child_key = child_key.strip()
                    child_value = child_value.strip()
                    index += 1
                    if child_value:
                        item[child_key] = _parse_scalar(child_value)
                    else:
                        item[child_key] = parse_block(expected_indent + 4)

                result.append(item)
            else:
                result.append(_parse_scalar(payload))

        return result

    def parse_dict(expected_indent: int) -> dict[str, Any]:
        nonlocal index
        result: dict[str, Any] = {}

        while index < len(lines):
            indent, text = lines[index]
            if indent < expected_indent:
                break
            if indent != expected_indent:
                raise ValueError(f"非法缩进: {text}")
            if text.startswith("-"):
                break

            if ":" not in text:
                raise ValueError(f"非法 key/value: {text}")
            key, value_str = text.split(":", 1)
            key = key.strip()
            value_str = value_str.strip()
            index += 1

            if value_str:
                result[key] = _parse_scalar(value_str)
            else:
                result[key] = parse_block(expected_indent + 2)

        return result

    return parse_block(lines[0][0])


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]

    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value