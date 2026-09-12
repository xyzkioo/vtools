"""轻量、可读的模型版本与实验分支注册表。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


def artifact_info(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"模型产物不存在：{target}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(target), "size_bytes": target.stat().st_size, "sha256": digest.hexdigest()}


def environment_info() -> dict[str, Any]:
    """记录可复现实验所需的轻量环境信息，不强制导入 PyTorch。"""

    info: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch  # type: ignore

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
    except ImportError:
        info["torch"] = None
    return info


class ModelRegistry:
    """使用单个 JSON 文件保存项目、分支、版本和运行血缘。"""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._read()

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": 1, "branches": {}, "versions": {}, "relations": [], "runs": []}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"注册表必须是 JSON 对象：{self.path}")
        value.setdefault("schema_version", 1)
        value.setdefault("branches", {})
        value.setdefault("versions", {})
        value.setdefault("relations", [])
        value.setdefault("runs", [])
        return value

    def save(self) -> None:
        payload = json.dumps(self.data, ensure_ascii=False, indent=2)
        fd, temporary = tempfile.mkstemp(prefix="registry.", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def ensure_branch(self, name: str, *, from_version: str | None = None, description: str = "") -> dict[str, Any]:
        branch_name = str(name).strip()
        if not branch_name:
            raise ValueError("分支名称不能为空")
        existing = self.data["branches"].get(branch_name)
        if existing:
            if from_version and existing.get("head_version_id") is None:
                if from_version not in self.data["versions"]:
                    raise KeyError(f"找不到分支起点模型版本：{from_version}")
                existing["head_version_id"] = from_version
                existing["updated_at"] = datetime.now(timezone.utc).isoformat()
                self.data["branches"][branch_name] = existing
                self.save()
            elif from_version and existing.get("head_version_id") != from_version:
                raise ValueError(f"分支已存在且指针不同：{branch_name}")
            return dict(existing)
        if from_version and from_version not in self.data["versions"]:
            raise KeyError(f"找不到分支起点模型版本：{from_version}")
        branch = {
            "name": branch_name,
            "description": description,
            "head_version_id": from_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.data["branches"][branch_name] = branch
        self.save()
        return dict(branch)

    def get_branch(self, name: str) -> dict[str, Any]:
        try:
            return dict(self.data["branches"][name])
        except KeyError as exc:
            raise KeyError(f"找不到分支：{name}") from exc

    def add_version(
        self,
        *,
        name: str,
        artifact: str | Path,
        parent_version_id: str | None,
        run_id: str,
        method: str,
        metadata: Optional[Mapping[str, Any]] = None,
        teacher_version_id: str | None = None,
        student_init_version_id: str | None = None,
    ) -> dict[str, Any]:
        import uuid

        version_id = f"v-{uuid.uuid4().hex[:12]}"
        record = {
            "id": version_id,
            "name": name,
            "artifact": artifact_info(artifact),
            "parent_version_id": parent_version_id,
            "teacher_version_id": teacher_version_id,
            "student_init_version_id": student_init_version_id,
            "run_id": run_id,
            "method": method,
            "metadata": dict(metadata or {}),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.data["versions"][version_id] = record
        if parent_version_id:
            self.data["relations"].append({"source_version_id": parent_version_id, "target_version_id": version_id, "role": "parent"})
        if teacher_version_id:
            self.data["relations"].append({"source_version_id": teacher_version_id, "target_version_id": version_id, "role": "teacher"})
        if student_init_version_id:
            self.data["relations"].append({"source_version_id": student_init_version_id, "target_version_id": version_id, "role": "student_initialization"})
        self.save()
        return dict(record)

    def advance_branch(self, branch_name: str, version_id: str) -> dict[str, Any]:
        if version_id not in self.data["versions"]:
            raise KeyError(f"找不到模型版本：{version_id}")
        branch = self.get_branch(branch_name)
        branch["head_version_id"] = version_id
        branch["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.data["branches"][branch_name] = branch
        self.save()
        return dict(branch)

    def add_run(self, run_id: str, record: Mapping[str, Any]) -> None:
        value = {"id": run_id, **dict(record)}
        for index, existing in enumerate(self.data["runs"]):
            if isinstance(existing, Mapping) and existing.get("id") == run_id:
                self.data["runs"][index] = {**dict(existing), **value}
                self.save()
                return
        self.data["runs"].append(value)
        self.save()

    def lineage(self, version_id: str | None) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        current = version_id
        visited: set[str] = set()
        while current:
            if current in visited:
                raise ValueError("检测到循环模型血缘")
            visited.add(current)
            version = self.data["versions"].get(current)
            if version is None:
                raise KeyError(f"找不到模型版本：{current}")
            result.append(dict(version))
            current = version.get("parent_version_id")
        return result

    def versions_for_branch(self, branch_name: str) -> list[dict[str, Any]]:
        branch = self.get_branch(branch_name)
        return self.lineage(branch.get("head_version_id"))


__all__ = ["ModelRegistry", "artifact_info", "environment_info"]
