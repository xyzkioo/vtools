# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import re
import shutil
import subprocess
import tarfile
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from urllib import parse
from uuid import uuid4

from ultralytics.utils import ASSETS_URL, LOGGER, TQDM, checks, clean_url, emojis, is_online, url2file

# 定义由 https://github.com/ultralytics/assets 维护的 Ultralytics GitHub 资源
GITHUB_ASSETS_REPO = "ultralytics/assets"
GITHUB_ASSETS_NAMES = frozenset(
    [f"yolov8{k}{suffix}.pt" for k in "nsmlx" for suffix in ("", "-cls", "-seg", "-pose", "-obb", "-oiv7")]
    + [f"yolo11{k}{suffix}.pt" for k in "nsmlx" for suffix in ("", "-cls", "-seg", "-pose", "-obb")]
    + [f"yolo12{k}{suffix}.pt" for k in "nsmlx" for suffix in ("",)]  # 当前仅包含检测模型
    + [f"yolo26{k}{suffix}.pt" for k in "nsmlx" for suffix in ("", "-cls", "-seg", "-sem", "-pose", "-obb", "-depth")]
    + [f"yolo26{k}-objv1{suffix}.pt" for k in "nsmlx" for suffix in ("-150", "-seg")]
    + [f"yolov5{k}{resolution}u.pt" for k in "nsmlx" for resolution in ("", "6")]
    + [f"yolov3{k}u.pt" for k in ("", "-spp", "-tiny")]
    + [f"yolov8{k}-world.pt" for k in "smlx"]
    + [f"yolov8{k}-worldv2.pt" for k in "smlx"]
    + [f"yoloe-v8{k}{suffix}.pt" for k in "sml" for suffix in ("-seg", "-seg-pf")]
    + [f"yoloe-11{k}{suffix}.pt" for k in "sml" for suffix in ("-seg", "-seg-pf")]
    + [f"yoloe-26{k}{suffix}.pt" for k in "nsmlx" for suffix in ("-seg", "-seg-pf")]
    + [f"yolov9{k}.pt" for k in "tsmce"]
    + [f"yolov10{k}.pt" for k in "nsmblx"]
    + [f"yolo_nas_{k}.pt" for k in "sml"]
    + [f"sam_{k}.pt" for k in "bl"]
    + [f"sam2_{k}.pt" for k in "blst"]
    + [f"sam2.1_{k}.pt" for k in "blst"]
    + [f"FastSAM-{k}.pt" for k in "sx"]
    + [f"rtdetr-{k}.pt" for k in "lx"]
    + [
        "mobile_sam.pt",
        "mobileclip_blt.ts",
        "yolo11n-grayscale.pt",
        "calibration_image_sample_data_20x128x128x3_float32.npy.zip",
    ]
)
GITHUB_ASSETS_STEMS = frozenset(k.rpartition(".")[0] for k in GITHUB_ASSETS_NAMES)


def is_url(url: str | Path, check: bool = False) -> bool:
    """验证给定字符串是否为 URL，并可选地检查该 URL 是否在线存在。

    参数：
        url (str | Path): 要验证为 URL 的字符串。
        check (bool, 可选): 为 True 时额外检查 URL 是否在线存在。

    返回：
        (bool): URL 有效时返回 True；当 `check` 为 True 时，只有 URL 在线存在才返回 True。

    示例：
        >>> valid = is_url("https://www.example.com")
        >>> valid_and_exists = is_url("https://www.example.com", check=True)
    """
    try:
        url = str(url)
        result = parse.urlparse(url)
        if not (result.scheme and result.netloc):
            return False
        if check:
            import requests  # scoped as slow import

            return requests.head(url, timeout=3, allow_redirects=True).ok
        return True
    except Exception:
        return False


def delete_dsstore(path: str | Path, files_to_delete: tuple[str, ...] = (".DS_Store", "__MACOSX")) -> None:
    """删除目录中指定的所有系统文件和目录。

    参数：
        path (str | Path): 要删除文件的目录路径。
        files_to_delete (tuple[str, ...]): 要删除的文件和目录名称。

    示例：
        >>> from ultralytics.utils.downloads import delete_dsstore
        >>> delete_dsstore("path/to/dir")

    注意：
        `.DS_Store` 文件由 Apple 操作系统创建，包含文件夹和文件的元数据。这些隐藏系统文件在不同操作系统之间
        传输文件时可能造成问题。
    """
    for file in files_to_delete:
        matches = sorted(Path(path).rglob(file), key=lambda x: len(x.parts), reverse=True)
        LOGGER.info(f"Deleting {file} files: {matches}")
        for f in matches:
            if f.is_dir() and not f.is_symlink():
                shutil.rmtree(f)
            else:
                f.unlink()


def zip_directory(
    directory: str | Path,
    compress: bool = True,
    exclude: tuple[str, ...] = (".DS_Store", "__MACOSX"),
    progress: bool = True,
) -> Path:
    """压缩目录内容，并排除指定文件。

    生成的 ZIP 文件以目录命名，并放置在该目录旁边。

    参数：
        directory (str | Path): 要压缩的目录路径。
        compress (bool): 压缩时是否压缩文件内容。
        exclude (tuple[str, ...], 可选): 要排除的文件名字符串元组。
        progress (bool, 可选): 是否显示进度条。

    返回：
        (Path): 生成的 ZIP 文件路径。

    示例：
        >>> from ultralytics.utils.downloads import zip_directory
        >>> file = zip_directory("path/to/dir")
    """
    from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

    delete_dsstore(directory)
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    # 使用进度条压缩
    files = [f for f in directory.rglob("*") if f.is_file() and all(x not in f.name for x in exclude)]  # 待压缩文件
    zip_file = directory.with_suffix(".zip")
    compression = ZIP_DEFLATED if compress else ZIP_STORED
    with ZipFile(zip_file, "w", compression) as f:
        for file in TQDM(files, desc=f"Zipping {directory} to {zip_file}...", unit="files", disable=not progress):
            f.write(file, file.relative_to(directory))

    return zip_file  # 返回 ZIP 文件路径


def unzip_file(
    file: str | Path,
    path: str | Path | None = None,
    exclude: tuple[str, ...] = (".DS_Store", "__MACOSX"),
    exist_ok: bool = False,
    progress: bool = True,
) -> Path:
    """将 `*.zip` 文件解压到指定路径，并排除指定文件。

    如果 ZIP 文件不包含单个顶层目录，则创建一个与 ZIP 文件同名（不含扩展名）的新目录来解压内容。
    如果未提供路径，则使用 ZIP 文件的父目录作为默认路径。

    参数：
        file (str | Path): 要解压的 ZIP 文件路径。
        path (str | Path, 可选): ZIP 文件的解压路径。
        exclude (tuple[str, ...], 可选): 要排除的文件名字符串元组。
        exist_ok (bool, 可选): 目标内容已存在时是否覆盖。
        progress (bool, 可选): 是否显示进度条。

    返回：
        (Path): ZIP 文件解压后的目录路径。

    异常：
        BadZipFile: 给定文件不存在或不是有效 ZIP 文件时抛出。

    示例：
        >>> from ultralytics.utils.downloads import unzip_file
        >>> directory = unzip_file("path/to/file.zip")
    """
    from zipfile import BadZipFile, ZipFile, is_zipfile

    if not (Path(file).exists() and is_zipfile(file)):
        raise BadZipFile(f"File '{file}' does not exist or is a bad zip file.")
    if path is None:
        path = Path(file).parent  # 默认路径

    # 解压文件内容
    with ZipFile(file) as zipObj:
        files = [f for f in zipObj.namelist() if all(x not in f for x in exclude)]
        top_level_dirs = {Path(f).parts[0] for f in files}

        # 决定直接解压，还是解压到新目录
        unzip_as_dir = len(top_level_dirs) == 1  # (len(文件) > 1 and not 文件[0].endswith("/"))
        if unzip_as_dir:
            # ZIP 包含一个顶层目录
            extract_path = path  # 例如 ../datasets
            path = Path(path) / next(iter(top_level_dirs))  # 例如将 coco8/ 解压到 ../datasets/
        else:
            # ZIP 顶层包含多个文件
            path = extract_path = Path(path) / Path(file).stem  # 例如将多个文件解压到 ../datasets/coco8/

        # 检查目标目录是否已存在且包含文件
        if path.exists() and any(path.iterdir()) and not exist_ok:
            # 如果目标目录已存在且非空，则直接返回路径而不解压
            LOGGER.warning(f"Skipping {file} unzip as destination directory {path} is not empty.")
            return path

        extract_path = Path(extract_path).resolve()
        for f in TQDM(files, desc=f"Unzipping {file} to {Path(path).resolve()}...", unit="files", disable=not progress):
            f_path = Path(f)
            target = (extract_path / f_path).resolve()
            if (
                f_path.is_absolute()
                or ".." in f_path.parts
                or target.parts[: len(extract_path.parts)] != extract_path.parts
            ):
                LOGGER.warning(f"Potentially insecure file path: {f}, skipping extraction.")
                continue
            zipObj.extract(f, extract_path)

    return path  # 返回解压目录


def check_disk_space(
    file_bytes: int,
    path: str | Path | None = None,
    sf: float = 1.5,
    hard: bool = True,
) -> bool:
    """检查磁盘空间是否足以下载并保存文件。

    参数：
        file_bytes (int): 文件大小，单位为字节。
        path (str | Path, 可选): 要检查可用空间的路径或驱动器。
        sf (float, 可选): 安全系数，用于乘以所需可用空间。
        hard (bool, 可选): 磁盘空间不足时是否抛出错误。

    返回：
        (bool): 磁盘空间充足时返回 True，否则返回 False。
    """
    total, _used, free = shutil.disk_usage(path or Path.cwd())  # bytes
    # 无法报告使用情况的文件系统会返回 0 个总块；对于有效总容量，free == 0 确实表示磁盘已满，仍必须捕获，
    # 因为 `free` 统计的是非特权进程可用的块数。
    if not total or file_bytes * sf < free:
        return True  # 空间充足

    def fmt_bytes(b):
        if b < (1 << 20):  # 如果没有 KB 档位，低于 1 MB 的值都会显示为 0.0 MB，无法反映磁盘空间状态
            return f"{b / (1 << 10):.1f} KB"
        return f"{b / (1 << 20):.1f} MB" if b < (1 << 30) else f"{b / (1 << 30):.3f} GB"

    # 空间不足
    text = (
        f"Insufficient free disk space {fmt_bytes(free)} < {fmt_bytes(int(file_bytes * sf))} required, "
        f"Please free {fmt_bytes(int(file_bytes * sf - free))} additional disk space and try again."
    )
    if hard:
        raise MemoryError(text)
    LOGGER.warning(text)
    return False


def get_google_drive_file_info(link: str) -> tuple[str, str | None]:
    """获取可共享 Google Drive 文件链接对应的直接下载地址和文件名。

    参数：
        link (str): Google Drive 文件的可共享链接。

    返回：
        url (str): Google Drive 文件的直接下载 URL。
        filename (str | None): Google Drive 文件的原始文件名；提取失败时返回 None。

    示例：
        >>> from ultralytics.utils.downloads import get_google_drive_file_info
        >>> link = "https://drive.google.com/file/d/1cqT-cJgANNrhIHCrEufUYhQ4RqiWG_lJ/view?usp=drive_link"
        >>> url, filename = get_google_drive_file_info(link)
    """
    import requests  # scoped as slow import

    file_id = link.split("/d/")[1].split("/view", 1)[0]
    drive_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    filename = None

    # 开始会话
    with requests.Session() as session:
        response = session.get(drive_url, stream=True)
        if "quota exceeded" in str(response.content.lower()):
            raise ConnectionError(
                emojis(
                    f"❌  Google Drive file download quota exceeded. "
                    f"Please try again later or download this file manually at {link}."
                )
            )
        for k, v in response.cookies.items():
            if k.startswith("download_warning"):
                drive_url += f"&confirm={v}"  # v 是令牌
        if cd := response.headers.get("content-disposition"):
            filename = re.findall('filename="(.+)"', cd)[0]
    return drive_url, filename


def safe_download(
    url: str | Path,
    file: str | Path | None = None,
    dir: str | Path | None = None,
    unzip: bool = True,
    delete: bool = False,
    curl: bool = False,
    retry: int = 3,
    min_bytes: float = 1e0,
    exist_ok: bool = False,
    progress: bool = True,
) -> Path | str:
    """从 URL 下载文件，并支持重试、解压和删除已下载文件。通过验证 Content-Length 增强了部分下载检测能力。

    参数：
        url (str | Path): 要下载文件的 URL。
        file (str | Path, 可选): 下载后的文件名；未提供时使用 URL 中的同名文件名。
        dir (str | Path, 可选): 保存下载文件的目录；未提供时使用当前工作目录。
        unzip (bool, 可选): 是否解压下载的文件。
        delete (bool, 可选): 解压后是否删除下载文件。
        curl (bool, 可选): 是否使用 curl 命令行工具下载。
        retry (int, 可选): 下载失败时的重试次数。
        min_bytes (float, 可选): 判定下载成功所需的最小文件大小（字节）。
        exist_ok (bool, 可选): 解压时是否覆盖已有内容。
        progress (bool, 可选): 下载过程中是否显示进度条。

    返回：
        (Path | str): 下载文件或解压目录的路径。

    示例：
        >>> from ultralytics.utils.downloads import safe_download
        >>> link = "https://ultralytics.com/assets/bus.jpg"
        >>> path = safe_download(link)
    """
    url = str(url)
    if "://" not in url and Path(url).is_file():  # 本地文件路径（Windows Python<3.10 需要检查 '://'）
        f = Path(url)
    else:
        import requests  # scoped as slow import

        gdrive = url.startswith("https://drive.google.com/")  # 检查 URL 是否为 Google Drive 链接
        if gdrive:
            url, file = get_google_drive_file_info(url)
        url = url.replace(" ", "%20")  # 为兼容 curl 编码空格

        f = Path(dir or ".") / (file or url2file(url))  # 将 URL 转换为文件名
        if not f.is_file():  # URL 和文件均不存在
            uri = (url if gdrive else clean_url(url)).replace(ASSETS_URL, "https://ultralytics.com/assets")  # clean
            desc = f"Downloading {uri} to '{f}'"
            f.parent.mkdir(parents=True, exist_ok=True)  # 目录不存在时创建目录
            target = f
            f = target.with_name(f".{target.name}.{uuid4().hex}.part")  # 仅在验证尺寸后发布文件
            curl_installed = shutil.which("curl")
            expected_size = None  # 从 Content-Length 设置，用于验证 curl 重试结果
            for i in range(retry + 1):
                try:
                    if (curl or i > 0) and curl_installed:  # 使用 curl 下载并重试，然后继续
                        s = "sS" * (not progress)  # 静默模式
                        # 这是停滞时间限制，而不是总传输时间限制：如果 300 秒内速度低于 1 B/s，则终止连接，
                        # 避免非守护绘图线程等待字体下载时阻塞解释器退出。
                        args = ["--connect-timeout", "30", "--speed-limit", "1", "--speed-time", "300"]
                        r = subprocess.run(
                            ["curl", "-#", f"-{s}L", url, "-o", f, "--retry", "3", "-C", "-", *args], check=False
                        ).returncode
                        assert r == 0, f"Curl return value {r}"
                    else:  # 使用 requests 下载；超时限制连接时间和分块读取间隔，不限制总传输时间
                        with requests.get(
                            url, stream=True, headers={"Accept-Encoding": "identity"}, timeout=(30, 300)
                        ) as response:
                            response.raise_for_status()
                            expected_size = int(response.headers.get("Content-Length", 0))
                            if i == 0 and expected_size > 1048576:
                                check_disk_space(expected_size, path=f.parent)
                            buffer_size = max(8192, min(1048576, expected_size // 1000)) if expected_size else 8192
                            with TQDM(
                                total=expected_size,
                                desc=desc,
                                disable=not progress,
                                unit="B",
                                unit_scale=True,
                                unit_divisor=1024,
                            ) as pbar, open(f, "wb") as f_opened:
                                for data in response.iter_content(chunk_size=buffer_size):
                                    f_opened.write(data)
                                    pbar.update(len(data))

                    if f.exists():
                        file_size = f.stat().st_size
                        if file_size > min_bytes:
                            # 检查下载是否完成（仅在已知预期大小时）
                            if expected_size and file_size != expected_size:
                                LOGGER.warning(
                                    f"Partial download: {file_size}/{expected_size} bytes ({file_size / expected_size * 100:.1f}%)"
                                )
                            else:
                                f.replace(target)
                                f = target
                                break  # 成功
                        f.unlink()  # 删除部分下载文件
                except MemoryError:
                    raise  # 立即重新抛出；磁盘空间不足时重试没有意义
                except Exception as e:
                    # 仅在最终失败时处理：重试会通过 curl `-C -` 续传部分文件；如果留下该文件，上面的
                    # `not f.is_file()` 判断会永久将其误认为完整缓存。
                    if i == 0 and not is_online():
                        f.unlink(missing_ok=True)
                        raise ConnectionError(
                            emojis(f"❌  Download failure for {uri}. Environment may be offline.")
                        ) from e
                    elif i >= retry:
                        f.unlink(missing_ok=True)
                        raise ConnectionError(
                            emojis(f"❌  Download failure for {uri}. Retry limit reached. {e}")
                        ) from e
                    LOGGER.warning(f"Download failure, retrying {i + 1}/{retry} {uri}... {e}")
            else:  # 没有任何尝试执行到 `break`，说明所有下载都未通过大小验证并已删除
                raise ConnectionError(emojis(f"❌  Download failure for {uri}. Retry limit reached."))

    if unzip and f.exists() and f.suffix in {"", ".zip", ".tar", ".gz"}:
        from zipfile import is_zipfile

        unzip_dir = (dir or f.parent).resolve()  # 提供目录时解压到该目录，否则在原位置解压
        if is_zipfile(f):
            unzip_dir = unzip_file(file=f, path=unzip_dir, exist_ok=exist_ok, progress=progress)  # 解压
        elif f.suffix in {".tar", ".gz"}:
            LOGGER.info(f"Unzipping {f} to {unzip_dir}...")
            with tarfile.open(f, "r:*") as tar:
                for m in tar:
                    if not (m.isfile() or m.isdir()) or m.issym() or m.islnk():
                        LOGGER.warning(f"Potentially insecure tar member: {m.name}, skipping extraction.")
                        continue
                    m_path = Path(m.name)
                    target = (unzip_dir / m_path).resolve()
                    if (
                        m_path.is_absolute()
                        or ".." in m_path.parts
                        or target.parts[: len(unzip_dir.parts)] != unzip_dir.parts
                    ):
                        LOGGER.warning(f"Potentially insecure file path: {m.name}, skipping extraction.")
                        continue
                    if m.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif source := tar.extractfile(m):
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with source, open(target, "wb") as out:  # 'f' 是归档路径，稍后会删除
                            shutil.copyfileobj(source, out)
        if delete:
            f.unlink()  # 删除归档文件
        return unzip_dir
    return f


def get_github_assets(
    repo: str = "ultralytics/assets",
    version: str = "latest",
    retry: bool = False,
) -> tuple[str, list[str]]:
    """从 GitHub 仓库获取指定版本的标签和资源。

    未指定版本时，获取最新发行版资源。

    参数：
        repo (str, 可选): 格式为 `owner/repo` 的 GitHub 仓库。
        version (str, 可选): 要获取资源的发行版版本。
        retry (bool, 可选): 请求失败时是否重试。

    返回：
        tag (str): 发行版标签。
        assets (列表[str]): 资源名称列表。

    示例：
        >>> tag, assets = get_github_assets(repo="ultralytics/assets", version="latest")
    """
    import requests  # scoped as slow import

    if version != "latest":
        version = f"tags/{version}"  # 例如 tags/v6.2
    url = f"https://api.github.com/repos/{repo}/releases/{version}"
    attempts = 2 if retry else 1  # 对临时网络错误或非 200 响应重试一次
    for attempt in range(attempts):
        try:
            r = requests.get(url, timeout=30)  # github api
        except requests.exceptions.RequestException as e:
            if attempt < attempts - 1:
                continue  # 临时网络错误，重新尝试
            LOGGER.warning(f"GitHub assets check failure for {url}: {e}")
            return "", []
        if r.status_code == 200 or r.reason == "rate limit exceeded":  # 不要重试表示速率限制的 403
            break
    if r.status_code != 200:
        LOGGER.warning(f"GitHub assets check failure for {url}: {r.status_code} {r.reason}")
        return "", []
    data = r.json()
    return data["tag_name"], [x["name"] for x in data["assets"]]  # 标签和资源，例如 ['yolo26n.pt', 'yolo11s.pt', ...]


def attempt_download_asset(
    file: str | Path,
    repo: str = "ultralytics/assets",
    release: str = "v8.4.0",
    **kwargs,
) -> str:
    """本地找不到文件时，尝试从 GitHub 发行版资源中下载该文件。

    参数：
        file (str | Path): 要下载的文件名或文件路径。
        repo (str, 可选): 格式为 `owner/repo` 的 GitHub 仓库。
        release (str, 可选): 要下载的具体发行版版本。
        **kwargs (Any): 下载过程使用的其他关键字参数。

    返回：
        (str): 下载文件的路径。

    示例：
        >>> file_path = attempt_download_asset("yolo26n.pt", repo="ultralytics/assets", release="latest")
    """
    from ultralytics.utils import SETTINGS  # scoped for circular import

    # YOLOv3/5u 文件名更新
    file = str(file)
    file = checks.check_yolov5u_filename(file)
    file = Path(file.strip().replace("'", ""))
    if file.exists():
        return str(file)
    elif (SETTINGS["weights_dir"] / file).exists():
        return str(SETTINGS["weights_dir"] / file)
    else:
        # 指定了 URL
        name = Path(parse.unquote(str(file))).name  # 将 '%2F' 等编码解码为 '/'
        download_url = f"https://github.com/{repo}/releases/download"
        if str(file).startswith(("http:/", "https:/")):  # download
            url = str(file).replace(":/", "://")  # Pathlib turns :// -> :/
            file = url2file(name)  # 解析身份验证查询字符串
            if Path(file).is_file():
                LOGGER.info(f"Found {clean_url(url)} locally at {file}")  # 文件已存在
            else:
                safe_download(url=url, file=file, min_bytes=1e5, **kwargs)

        elif repo == GITHUB_ASSETS_REPO and name in GITHUB_ASSETS_NAMES:
            safe_download(url=f"{download_url}/{release}/{name}", file=file, min_bytes=1e5, **kwargs)

        else:
            tag, assets = get_github_assets(repo, release)
            if not assets:
                tag, assets = get_github_assets(repo)  # latest release
            if name in assets:
                safe_download(url=f"{download_url}/{tag}/{name}", file=file, min_bytes=1e5, **kwargs)

        return str(file)


def download(
    url: str | list[str] | Path,
    dir: Path | None = None,
    unzip: bool = True,
    delete: bool = False,
    curl: bool = False,
    threads: int = 1,
    retry: int = 3,
    exist_ok: bool = False,
) -> None:
    """将指定 URL 的文件下载到给定目录。

    指定多个线程时支持并发下载。

    参数：
        url (str | list[str] | Path): 要下载文件的 URL 或 URL 列表。
        dir (Path, 可选): 保存文件的目录。
        unzip (bool, 可选): 下载后是否解压文件。
        delete (bool, 可选): 解压后是否删除 ZIP 文件。
        curl (bool, 可选): 是否使用 curl 下载。
        threads (int, 可选): 并发下载使用的线程数。
        retry (int, 可选): 下载失败时的重试次数。
        exist_ok (bool, 可选): 解压时是否覆盖已有内容。

    示例：
        >>> download("https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg", dir="path/to/dir")
    """
    dir = Path(dir or Path.cwd())
    dir.mkdir(parents=True, exist_ok=True)  # 创建目录
    urls = [url] if isinstance(url, (str, Path)) else url
    if threads > 1:
        LOGGER.info(f"Downloading {len(urls)} file(s) with {threads} threads to {dir}...")
        with ThreadPool(threads) as pool:
            pool.map(
                lambda x: safe_download(
                    url=x[0],
                    dir=x[1],
                    unzip=unzip,
                    delete=delete,
                    curl=curl,
                    retry=retry,
                    exist_ok=exist_ok,
                    progress=True,
                ),
                zip(urls, repeat(dir)),
            )
            pool.close()
            pool.join()
    else:
        for u in urls:
            safe_download(url=u, dir=dir, unzip=unzip, delete=delete, curl=curl, retry=retry, exist_ok=exist_ok)
