"""旧版模型版本注册表 API。

压缩主入口不再实例化该注册表；每个 runN 在进程内维护连续处理状态。
该模块暂时保留，供已有调用方和历史 registry.json 读取代码平滑迁移。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import platform
import sys
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

try:
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - Windows
    fcntl = None
try:
    import msvcrt  # type: ignore
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None


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
        self._base_data = copy.deepcopy(self.data)

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

    @contextmanager
    def _lock(self):
        """Serialize registry writers while retaining atomic replacement."""
        lock_path = self.path.with_name(self.path.name + ".lock")
        lock_path.touch(exist_ok=True)
        with lock_path.open("r+") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            elif msvcrt is not None:
                if lock_path.stat().st_size == 0:
                    handle.write("0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                elif msvcrt is not None:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

    def _merge_disk(self, disk: dict[str, Any]) -> None:
        """Apply only this instance's changes onto the latest disk snapshot."""
        base = self._base_data
        merged = copy.deepcopy(disk)
        for key in ("branches", "versions"):
            local = self.data.get(key, {})
            old = base.get(key, {})
            target = merged.setdefault(key, {})
            for item_id, value in local.items():
                if old.get(item_id) != value:
                    previous = old.get(item_id)
                    current = target.get(item_id)
                    if item_id not in old and item_id in target and current != value:
                        raise RuntimeError(f"注册表并发创建冲突：{key}.{item_id} 已被其他写入者创建")
                    if item_id in old and item_id not in target:
                        raise RuntimeError(f"注册表并发更新冲突：{key}.{item_id} 已被其他写入者删除")
                    if isinstance(value, Mapping) and isinstance(previous, Mapping) and isinstance(current, Mapping):
                        updated = copy.deepcopy(current)
                        for field, field_value in value.items():
                            if previous.get(field) != field_value:
                                if current.get(field) != previous.get(field) and current.get(field) != field_value:
                                    raise RuntimeError(
                                        f"注册表并发更新冲突：{key}.{item_id}.{field} 已被其他写入者修改"
                                    )
                                updated[field] = copy.deepcopy(field_value)
                        for field in set(previous) - set(value):
                            if previous.get(field) == updated.get(field):
                                updated.pop(field, None)
                        target[item_id] = updated
                    else:
                        target[item_id] = copy.deepcopy(value)
            for item_id in set(old) - set(local):
                if old.get(item_id) == target.get(item_id):
                    target.pop(item_id, None)

        # Runs are keyed by id. A later status update replaces the old record;
        # distinct run IDs are appended once.
        old_runs = {str(item.get("id")): item for item in base.get("runs", []) if isinstance(item, Mapping)}
        local_runs = {str(item.get("id")): item for item in self.data.get("runs", []) if isinstance(item, Mapping)}
        disk_runs = {str(item.get("id")): item for item in merged.get("runs", []) if isinstance(item, Mapping)}
        for run_id, value in local_runs.items():
            if old_runs.get(run_id) != value:
                disk_runs[run_id] = copy.deepcopy(value)
        merged["runs"] = list(disk_runs.values())

        old_relations = base.get("relations", [])
        local_relations = self.data.get("relations", [])
        if local_relations != old_relations:
            relations = list(merged.get("relations", []))
            seen = {json.dumps(item, sort_keys=True, ensure_ascii=False) for item in relations}
            for item in local_relations:
                marker = json.dumps(item, sort_keys=True, ensure_ascii=False)
                if marker not in seen:
                    relations.append(copy.deepcopy(item))
                    seen.add(marker)
            merged["relations"] = relations
        merged["schema_version"] = max(int(merged.get("schema_version", 1)), int(self.data.get("schema_version", 1)))
        self.data = merged

    def save(self) -> None:
        with self._lock():
            if self.path.exists():
                self._merge_disk(self._read())
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
            self._base_data = copy.deepcopy(self.data)

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
        value = {**dict(record), "id": run_id}
        for index, existing in enumerate(self.data["runs"]):
            if isinstance(existing, Mapping) and existing.get("id") == run_id:
                self.data["runs"][index] = {**dict(existing), **value}
                self.save()
                return
        self.data["runs"].append(value)
        self.save()

    def resolve_input_version(
        self,
        branch_name: str,
        explicit_weights: str | Path | None,
    ) -> tuple[str | None, str | None]:
        """解析一次运行真正要使用的模型输入。

        返回 ``(version_id, artifact_path)``：

        - 分支还没有版本时返回 ``(None, 显式权重路径)``；
        - 分支已有版本时始终返回 head 版本，因为连续的模块（例如
          剪枝 -> 导出）必须消费上一次推进后的产物；
        - ``model.weights`` 允许指向 head 或它的任一祖先（配置文件通常一直
          写着原始权重），但指向分支之外的权重会直接报错，避免把两份模型
          混在同一条血缘里。
        """

        head = self.get_branch(branch_name).get("head_version_id")
        resolved: str | None = None
        if explicit_weights:
            resolved = str(Path(str(explicit_weights)).expanduser().resolve())
        if not head:
            return None, resolved
        head_version = self.data["versions"][str(head)]
        head_path = str(head_version.get("artifact", {}).get("path") or "")
        if resolved is None:
            return str(head), head_path
        requested = artifact_info(resolved)
        known = {
            str(version.get("artifact", {}).get("sha256"))
            for version in self.lineage(str(head))
        }
        if requested.get("sha256") not in known:
            raise ValueError(
                "model.weights 与当前分支已有的模型版本不是同一份权重；"
                "请改成该分支的模型，或新建 branch.name 处理新的权重。"
            )
        return str(head), head_path

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
