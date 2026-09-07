#!/usr/bin/env python3
"""图片数据集批量改名工具。

支持：
1. 普通图片文件夹：只改图片文件名。
2. YOLO 数据集：同步修改 images/ 与 labels/ 中的同名 .txt 文件。
3. COCO 数据集：同步修改 COCO JSON 的 images[].file_name。

默认只预览；只有配置 apply=True 或传入 --apply 才会真正修改文件。
本文件顶部的 PYCHARM_CONFIG 可直接在 PyCharm 中修改后运行，不需要命令行参数。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


# ============================================================================
# PyCharm 直接运行配置：在这里修改后，点击 PyCharm 的运行按钮即可。
# dataset_format 可选："images"、"yolo"、"coco"、"auto"
#
# YOLO 示例：
#   input_dir: /home/用户名/datasets/flower/images
#   dataset_format: "yolo"
#   labels_dir: /home/用户名/datasets/flower/labels
#
# COCO 示例：
#   input_dir: /home/用户名/datasets/coco/images
#   dataset_format: "coco"
#   coco_json: /home/用户名/datasets/coco/annotations/instances_train.json
# ============================================================================
PYCHARM_CONFIG: dict[str, Any] = {
    "input_dir": r"/home/xyzkioo/PycharmProjects/vtools/flowerhard0",
    "dataset_format": "yolo",
    "labels_dir": r"",
    "coco_json": r"",
    "recursive": True,
    "include_hidden": False,
    "extensions": "jpg,jpeg,png,bmp,gif,tif,tiff,webp,heic,avif",
    "template": "{prefix}{index:04d}{suffix}",
    "prefix": "",
    "suffix": "",
    "start": 1,
    "digits": None,  # 例如填写 5，生成 00001、00002；填写 None 才使用 template
    "lowercase_ext": False,
    "allow_unreferenced": False,  # COCO 中未被 JSON 引用的图片是否也允许改名
    "apply": True,  # True=真正执行；False=只预览
    "yes": False,  # True=执行时跳过 yes 确认
}


DEFAULT_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
    ".heic",
    ".avif",
}


@dataclass(frozen=True)
class RenameItem:
    old: Path
    new: Path
    kind: str = "image"


@dataclass(frozen=True)
class CocoUpdate:
    record_index: int
    old_name: str
    new_name: str


def natural_key(path: Path) -> list[object]:
    """让 img2.jpg 排在 img10.jpg 前面。"""
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name)]


def parse_extensions(value: str) -> set[str]:
    extensions = set()
    for item in value.split(","):
        item = item.strip().lower()
        if item:
            extensions.add(item if item.startswith(".") else f".{item}")
    if not extensions:
        raise ValueError("图片扩展名不能为空")
    return extensions


def collect_images(directory: Path, extensions: set[str], recursive: bool, include_hidden: bool) -> list[Path]:
    iterator: Iterable[Path] = directory.rglob("*") if recursive else directory.iterdir()
    files = []
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if not include_hidden and any(part.startswith(".") for part in path.relative_to(directory).parts):
            continue
        files.append(path)
    return sorted(files, key=natural_key)


def render_name(
    template: str,
    index: int,
    path: Path,
    prefix: str,
    suffix: str,
    lowercase_ext: bool,
) -> str:
    extension = path.suffix.lower() if lowercase_ext else path.suffix
    values = {
        "index": index,
        "index0": index - 1,
        "stem": path.stem,
        "old_name": path.name,
        "ext": extension.lstrip("."),
        "prefix": prefix,
        "suffix": suffix,
    }
    try:
        base = template.format_map(values)
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError(
            f"模板错误：{exc}。可用变量：{{index}}、{{index0}}、{{stem}}、{{old_name}}、{{ext}}、{{prefix}}、{{suffix}}"
        ) from exc
    if not base.strip():
        raise ValueError("模板生成了空文件名")
    if base in {".", ".."} or "/" in base or "\\" in base:
        raise ValueError(f"模板生成了非法文件名：{base!r}。模板不能包含路径分隔符")
    return base if base.endswith(extension) else f"{base}{extension}"


def build_image_plan(
    files: list[Path],
    template: str,
    start: int,
    prefix: str,
    suffix: str,
    lowercase_ext: bool,
) -> list[RenameItem]:
    return [
        RenameItem(
            old=old,
            new=old.with_name(render_name(template, start + offset, old, prefix, suffix, lowercase_ext)),
            kind="image",
        )
        for offset, old in enumerate(files)
    ]


def resolve_yolo_roots(directory: Path, labels_dir: str | Path | None) -> tuple[Path, Path]:
    """允许 --dir 指向数据集根目录，也允许直接指向 images 目录。"""
    if labels_dir:
        image_root = directory
        label_root = Path(labels_dir).expanduser().resolve()
        if (directory / "images").is_dir() and directory.name.lower() not in {"images", "train", "val", "test"}:
            image_root = directory / "images"
        return image_root, label_root

    if directory.name.lower() == "images" and (directory.parent / "labels").is_dir():
        return directory, directory.parent / "labels"
    if (directory / "images").is_dir() and (directory / "labels").is_dir():
        return directory / "images", directory / "labels"
    raise ValueError(
        "YOLO 数据集目录未识别。请将 --dir 指向 images 文件夹，或同时提供 --labels-dir /数据集/labels"
    )


def build_yolo_label_plan(
    image_plan: list[RenameItem], image_root: Path, label_root: Path
) -> tuple[list[RenameItem], list[Path]]:
    label_plan: list[RenameItem] = []
    missing: list[Path] = []
    for image_item in image_plan:
        relative_old = image_item.old.relative_to(image_root)
        relative_new = image_item.new.relative_to(image_root)
        old_label = (label_root / relative_old).with_suffix(".txt")
        new_label = (label_root / relative_new).with_suffix(".txt")
        if old_label.exists():
            label_plan.append(RenameItem(old_label, new_label, "yolo-label"))
        else:
            missing.append(old_label)
    return label_plan, missing


def normalize_coco_name(value: str) -> str:
    return PurePosixPath(value.replace("\\", "/")).as_posix().lstrip("./")


def coco_candidate_keys(path: Path, image_root: Path, dataset_root: Path) -> set[str]:
    keys = {normalize_coco_name(path.name)}
    try:
        keys.add(normalize_coco_name(path.relative_to(image_root).as_posix()))
    except ValueError:
        pass
    try:
        keys.add(normalize_coco_name(path.relative_to(dataset_root).as_posix()))
    except ValueError:
        pass
    if image_root.name.lower() == "images":
        try:
            keys.add(normalize_coco_name((Path("images") / path.relative_to(image_root)).as_posix()))
        except ValueError:
            pass
    return keys


def load_coco_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not isinstance(data.get("images"), list):
        raise ValueError(f"不是有效的 COCO 标注文件：{path}（缺少 images 数组）")
    return data


def build_coco_updates(
    data: dict[str, Any], image_plan: list[RenameItem], image_root: Path, dataset_root: Path
) -> tuple[list[CocoUpdate], list[Path]]:
    by_key: dict[str, list[RenameItem]] = defaultdict(list)
    for item in image_plan:
        for key in coco_candidate_keys(item.old, image_root, dataset_root):
            by_key[key].append(item)

    updates: list[CocoUpdate] = []
    matched: set[Path] = set()
    for record_index, record in enumerate(data["images"]):
        if not isinstance(record, dict) or not isinstance(record.get("file_name"), str):
            continue
        file_name = record["file_name"]
        candidates: list[RenameItem] = []
        for key in {normalize_coco_name(file_name), normalize_coco_name(Path(file_name).name)}:
            candidates.extend(by_key.get(key, []))
        unique = {item.old: item for item in candidates}
        if len(unique) != 1:
            continue
        item = next(iter(unique.values()))
        matched.add(item.old)
        if item.old != item.new:
            new_name = PurePosixPath(file_name.replace("\\", "/")).with_name(item.new.name).as_posix()
            updates.append(CocoUpdate(record_index, file_name, new_name))

    unreferenced = [item.old for item in image_plan if item.old != item.new and item.old not in matched]
    return updates, unreferenced


def validate_plan(plan: list[RenameItem]) -> None:
    targets = [item.new for item in plan]
    duplicate_targets = {path for path in targets if targets.count(path) > 1}
    if duplicate_targets:
        names = ", ".join(sorted(str(path) for path in duplicate_targets))
        raise ValueError(f"目标文件名重复：{names}")

    source_set = {item.old for item in plan}
    for item in plan:
        if item.old == item.new:
            continue
        if item.new.exists() and item.new not in source_set:
            raise FileExistsError(f"目标已存在，为避免覆盖已停止：{item.new}")


def write_csv_plan(plan: list[RenameItem], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["类型", "原文件", "新文件"])
        writer.writerows([[item.kind, str(item.old), str(item.new)] for item in plan])


def apply_plan(plan: list[RenameItem]) -> int:
    """两阶段改名，支持文件名互换；失败时尽量完整回滚。"""
    changed = [item for item in plan if item.old != item.new]
    if not changed:
        return 0

    temporary: list[tuple[Path, Path, Path]] = []
    committed: list[tuple[Path, Path]] = []
    try:
        for item in changed:
            with tempfile.NamedTemporaryFile(
                prefix=".image_rename_", suffix=item.old.suffix, dir=item.old.parent, delete=False
            ) as handle:
                temp_path = Path(handle.name)
            temp_path.unlink()
            item.old.rename(temp_path)
            temporary.append((item.old, temp_path, item.new))
        for old_path, temp_path, new_path in temporary:
            temp_path.rename(new_path)
            committed.append((old_path, new_path))
    except Exception:
        for old_path, new_path in reversed(committed):
            if new_path.exists() and not old_path.exists():
                new_path.rename(old_path)
        for old_path, temp_path, _ in reversed(temporary):
            if temp_path.exists() and not old_path.exists():
                temp_path.rename(old_path)
        raise
    return len(changed)


def rollback_plan(plan: list[RenameItem]) -> None:
    changed = [item for item in plan if item.old != item.new]
    temporary: list[tuple[Path, Path]] = []
    for item in changed:
        if not item.new.exists():
            raise FileNotFoundError(f"回滚失败，找不到新文件：{item.new}")
        with tempfile.NamedTemporaryFile(
            prefix=".image_rollback_", suffix=item.new.suffix, dir=item.new.parent, delete=False
        ) as handle:
            temp_path = Path(handle.name)
        temp_path.unlink()
        item.new.rename(temp_path)
        temporary.append((item.old, temp_path))
    for old_path, temp_path in temporary:
        temp_path.rename(old_path)


def write_coco_with_backup(
    coco_path: Path, data: dict[str, Any], updates: list[CocoUpdate]
) -> Path | None:
    if not updates:
        return None
    updated = json.loads(json.dumps(data, ensure_ascii=False))
    for update in updates:
        updated["images"][update.record_index]["file_name"] = update.new_name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = coco_path.with_name(f"{coco_path.name}.{timestamp}.bak")
    coco_path.replace(backup)
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".json", dir=coco_path.parent, delete=False
        ) as handle:
            temp_json = Path(handle.name)
            json.dump(updated, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temp_json.replace(coco_path)
    except Exception:
        if coco_path.exists():
            coco_path.unlink()
        backup.replace(coco_path)
        raise
    return backup


def print_plan(plan: list[RenameItem], limit: int = 40) -> None:
    changed = [item for item in plan if item.old != item.new]
    image_count = sum(item.kind == "image" for item in plan)
    label_count = sum(item.kind == "yolo-label" for item in plan)
    print(f"共发现 {image_count} 张图片，关联标签 {label_count} 个，预计修改 {len(changed)} 个文件。")
    for item in plan[:limit]:
        marker = "=" if item.old == item.new else "→"
        kind = "图片" if item.kind == "image" else "标签"
        print(f"  [{kind}] {item.old} {marker} {item.new.name}")
    if len(plan) > limit:
        print(f"  ... 其余 {len(plan) - limit} 项省略，可用 --plan 导出完整清单。")


def config_or_arg(args: argparse.Namespace, name: str) -> Any:
    value = getattr(args, name)
    config_name = "input_dir" if name == "dir" else name
    return PYCHARM_CONFIG[config_name] if value is None else value


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量修改图片、YOLO 标签或 COCO JSON 中的文件名。")
    parser.add_argument("--dir", type=Path, default=None, help="图片目录；YOLO 也可填写数据集根目录")
    parser.add_argument("--dataset-format", choices=["images", "yolo", "coco", "auto"], default=None, help="数据集格式")
    parser.add_argument("--labels-dir", default=None, help="YOLO 标签目录；不填时自动识别同级 labels")
    parser.add_argument("--coco-json", type=Path, default=None, help="COCO 标注 JSON 文件")
    parser.add_argument("--recursive", dest="recursive", action="store_true", default=None, help="递归处理子目录")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="不处理子目录")
    parser.add_argument("--include-hidden", dest="include_hidden", action="store_true", default=None, help="包含隐藏路径")
    parser.add_argument("--extensions", default=None, help="图片扩展名，逗号分隔，例如 jpg,png,webp")
    parser.add_argument("--template", default=None, help="模板，例如 {index:04d}_{stem}")
    parser.add_argument("--prefix", default=None, help="模板中的 {prefix}")
    parser.add_argument("--suffix", default=None, help="模板中的 {suffix}")
    parser.add_argument("--start", type=int, default=None, help="编号起始值")
    parser.add_argument("--digits", type=int, default=None, help="编号位数，例如 4 生成 0001")
    parser.add_argument("--lowercase-ext", dest="lowercase_ext", action="store_true", default=None, help="扩展名统一为小写")
    parser.add_argument("--allow-unreferenced", dest="allow_unreferenced", action="store_true", default=None, help="允许 COCO 中未引用的图片继续改名")
    parser.add_argument("--plan", type=Path, default=None, help="导出 CSV 改名清单")
    parser.add_argument("--apply", dest="apply", action="store_true", default=None, help="真正执行修改")
    parser.add_argument("--dry-run", dest="apply", action="store_false", help="强制只预览")
    parser.add_argument("--yes", dest="yes", action="store_true", default=None, help="跳过执行确认")
    return parser


def main() -> int:
    args = create_parser().parse_args()
    try:
        raw_dir = config_or_arg(args, "dir")
        if not raw_dir:
            raise ValueError("请先在文件顶部 PYCHARM_CONFIG['input_dir'] 填写数据集路径，或使用 --dir")
        directory = Path(raw_dir).expanduser().resolve()
        if not directory.is_dir():
            raise NotADirectoryError(f"文件夹不存在：{directory}")

        dataset_format = config_or_arg(args, "dataset_format").lower()
        labels_dir = config_or_arg(args, "labels_dir")
        coco_json_value = config_or_arg(args, "coco_json")
        recursive = config_or_arg(args, "recursive")
        include_hidden = config_or_arg(args, "include_hidden")
        extensions = parse_extensions(config_or_arg(args, "extensions"))
        template = config_or_arg(args, "template")
        prefix = config_or_arg(args, "prefix")
        suffix = config_or_arg(args, "suffix")
        start = config_or_arg(args, "start")
        digits = config_or_arg(args, "digits")
        lowercase_ext = config_or_arg(args, "lowercase_ext")
        allow_unreferenced = config_or_arg(args, "allow_unreferenced")
        apply = config_or_arg(args, "apply")
        yes = config_or_arg(args, "yes")

        if dataset_format == "auto":
            if coco_json_value:
                dataset_format = "coco"
            elif labels_dir or (directory.name.lower() == "images" and (directory.parent / "labels").is_dir()):
                dataset_format = "yolo"
            else:
                dataset_format = "images"
        if start < 0:
            raise ValueError("编号起始值不能小于 0")
        if digits is not None:
            if digits < 1:
                raise ValueError("编号位数必须大于 0")
            template = f"{{prefix}}{{index:0{digits}d}}{{suffix}}"

        image_root = directory
        label_root: Path | None = None
        if dataset_format == "yolo":
            image_root, label_root = resolve_yolo_roots(directory, labels_dir)
        elif dataset_format == "coco" and (directory / "images").is_dir():
            image_root = directory / "images"
        elif dataset_format not in {"images", "coco"}:
            raise ValueError(f"不支持的数据集格式：{dataset_format}")

        files = collect_images(image_root, extensions, recursive, include_hidden)
        if not files:
            print(f"没有在 {image_root} 找到符合条件的图片。")
            return 0
        image_plan = build_image_plan(files, template, start, prefix, suffix, lowercase_ext)
        all_plan = list(image_plan)
        missing_labels: list[Path] = []
        coco_updates: list[CocoUpdate] = []
        coco_data: dict[str, Any] | None = None
        coco_path: Path | None = None
        unreferenced: list[Path] = []

        if dataset_format == "yolo":
            assert label_root is not None
            label_plan, missing_labels = build_yolo_label_plan(image_plan, image_root, label_root)
            all_plan.extend(label_plan)
        elif dataset_format == "coco":
            if not coco_json_value:
                raise ValueError("COCO 格式必须填写 coco_json 或使用 --coco-json")
            coco_path = Path(coco_json_value).expanduser()
            if not coco_path.is_absolute():
                coco_path = (directory / coco_path).resolve()
            if not coco_path.is_file():
                raise FileNotFoundError(f"COCO JSON 不存在：{coco_path}")
            coco_data = load_coco_json(coco_path)
            coco_updates, unreferenced = build_coco_updates(coco_data, image_plan, image_root, image_root.parent)
            if unreferenced and not allow_unreferenced:
                examples = ", ".join(str(path) for path in unreferenced[:3])
                raise ValueError(
                    f"有 {len(unreferenced)} 张图片没有在 COCO JSON 中匹配到。示例：{examples}；"
                    "为避免 JSON 与图片不一致，默认停止。确认后可加 --allow-unreferenced。"
                )

        validate_plan(all_plan)
        print(f"数据集格式：{dataset_format}；图片目录：{image_root}")
        print_plan(all_plan)
        if missing_labels:
            print(f"提示：{len(missing_labels)} 张图片没有对应 YOLO 标签，已跳过不存在的标签文件。")
        if dataset_format == "coco":
            print(f"COCO JSON 将更新 {len(coco_updates)} 条 images.file_name 记录。")
            if unreferenced:
                print(f"提示：有 {len(unreferenced)} 张图片未被 COCO JSON 引用，但已按 --allow-unreferenced 处理。")
        if args.plan:
            plan_path = args.plan.expanduser().resolve()
            write_csv_plan(all_plan, plan_path)
            print(f"完整改名清单已保存：{plan_path}")

        changed = sum(item.old != item.new for item in all_plan)
        if not apply:
            print("当前为预览模式。确认无误后，在 PyCharm 配置 apply=True，或命令末尾加 --apply。")
            return 0
        if changed == 0 and not coco_updates:
            print("没有需要修改的文件名或标注记录。")
            return 0
        if not yes:
            answer = input(f"确认修改 {changed} 个文件，并更新 {len(coco_updates)} 条标注记录？输入 yes 继续：").strip().lower()
            if answer != "yes":
                print("已取消，没有修改文件。")
                return 0

        apply_plan(all_plan)
        backup: Path | None = None
        try:
            if dataset_format == "coco" and coco_path and coco_data is not None:
                backup = write_coco_with_backup(coco_path, coco_data, coco_updates)
        except Exception:
            rollback_plan(all_plan)
            raise
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"已完成：{changed} 个文件名（{timestamp}）。")
        if backup:
            print(f"COCO 原 JSON 已备份为：{backup}")
        return 0
    except (ValueError, FileExistsError, FileNotFoundError, NotADirectoryError, OSError, json.JSONDecodeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
