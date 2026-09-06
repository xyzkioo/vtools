# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import zlib
from functools import cached_property
from pathlib import Path


class GitRepo:
    """表示本地 Git 仓库，并提供分支、提交和远程仓库元数据。

    此类从给定路径向上查找 .git 条目来发现仓库根目录，解析实际的 .git 目录（包括 worktree），
    并直接读取磁盘上的 Git 元数据，不调用 git 可执行文件，因此可在受限环境中工作。所有元数据属性
    都采用延迟方式解析并缓存；如需刷新状态，请重新创建实例。

    属性：
        root (Path | None): 包含 .git 条目的仓库根目录；不在仓库中时为 None。
        gitdir (Path | None): 解析后的 .git 目录路径；支持 worktree；无法解析时为 None。
        refdir (Path | None): 包含共享引用、对象和配置的目录；无法解析时为 None。
        head (str | None): HEAD 的原始内容；分离 HEAD 时为 SHA，分支 HEAD 时为 "ref: <refname>"。
        is_repo (bool): 给定路径是否位于 Git 仓库中。
        branch (str | None): HEAD 指向分支时的当前分支名称；分离 HEAD 或不在仓库中时为 None。
        commit (str | None): HEAD 对应的当前提交 SHA；无法确定时为 None。
        message (str | None): 松散对象中的当前提交主题；无法确定时为 None。
        origin (str | None): 从 gitdir/config 读取的 "origin" 远程 URL；未设置或不可用时为 None。

    示例：
        从当前工作目录初始化对象并读取元数据
        >>> from pathlib import Path
        >>> repo = GitRepo(Path.cwd())
        >>> is_repo = repo.is_repo
        >>> branch, commit, origin = repo.branch, repo.commit, repo.origin

    注意：
        - 通过读取 HEAD、packed-refs、config 和对象文件解析元数据，不调用子进程。
        - 首次访问属性时使用 cached_property 缓存；如需反映仓库变化，请重新创建对象。
    """

    def __init__(self, path: Path | None = None):
        """从起始路径发现仓库根目录，并初始化 Git 仓库上下文。

        参数：
            路径 (Path, 可选): 用于定位仓库根目录的起始文件或目录路径。
        """
        self.root = self._find_root(path or Path(__file__).resolve())
        self.gitdir = self._gitdir(self.root) if self.root else None
        self.refdir = self._refdir(self.gitdir)

    @staticmethod
    def _find_root(p: Path) -> Path | None:
        """返回仓库根目录；如果不在仓库中则返回 None。"""
        return next((d for d in [p, *list(p.parents)] if (d / ".git").exists()), None)

    @staticmethod
    def _gitdir(root: Path) -> Path | None:
        """解析实际的 .git 目录（支持 worktree）。"""
        g = root / ".git"
        if g.is_dir():
            return g
        if g.is_file():
            t = g.read_text(errors="ignore").strip()
            if t.startswith("gitdir:"):
                return (root / t.split(":", 1)[1].strip()).resolve()
        return None

    @staticmethod
    def _refdir(gitdir: Path | None) -> Path | None:
        """解析包含 refs、对象和配置的目录。"""
        p = gitdir / "commondir" if gitdir else None
        if s := GitRepo._read(p):
            d = Path(s)
            return (gitdir / d).resolve() if not d.is_absolute() else d
        return gitdir

    @staticmethod
    def _read(p: Path | None) -> str | None:
        """如果文件存在，则读取并去除首尾空白。"""
        return p.read_text(errors="ignore").strip() if p and p.exists() else None

    @cached_property
    def head(self) -> str | None:
        """返回 HEAD 文件内容。"""
        return self._read(self.gitdir / "HEAD" if self.gitdir else None)

    def _ref_commit(self, ref: str) -> str | None:
        """返回引用对应的提交（支持 packed-refs）。"""
        rf = self.refdir / ref
        if s := self._read(rf):
            return s
        pf = self.refdir / "packed-refs"
        b = pf.read_bytes().splitlines() if pf.exists() else []
        tgt = ref.encode()
        for line in b:
            if line[:1] in (b"#", b"^") or b" " not in line:
                continue
            sha, name = line.split(b" ", 1)
            if name.strip() == tgt:
                return sha.decode()
        return None

    def _commit_subject(self, commit: str) -> str | None:
        """从松散对象读取提交主题；找不到时返回 None。"""
        obj = self.refdir / "objects" / commit[:2] / commit[2:]
        if not obj.exists():
            return None
        data = zlib.decompress(obj.read_bytes())
        if b"\0" not in data:
            return None
        kind, body = data.split(b"\0", 1)
        if not kind.startswith(b"commit ") or b"\n\n" not in body:
            return None
        subject = body.split(b"\n\n", 1)[1].splitlines()[0].decode(errors="replace").strip()
        return subject or None

    @property
    def is_repo(self) -> bool:
        """如果当前位于 Git 仓库中则返回 True。"""
        return self.gitdir is not None

    @cached_property
    def branch(self) -> str | None:
        """返回当前分支；如果无法确定则返回 None。"""
        if not self.is_repo or not self.head or not self.head.startswith("ref: "):
            return None
        ref = self.head[5:].strip()
        return ref[11:] if ref.startswith("refs/heads/") else ref

    @cached_property
    def commit(self) -> str | None:
        """返回当前提交 SHA；如果无法确定则返回 None。"""
        if not self.is_repo or not self.head:
            return None
        return self._ref_commit(self.head[5:].strip()) if self.head.startswith("ref: ") else self.head

    @cached_property
    def message(self) -> str | None:
        """返回当前提交主题；如果无法确定则返回 None。"""
        if not self.is_repo or not self.commit:
            return None
        return self._commit_subject(self.commit)

    @cached_property
    def origin(self) -> str | None:
        """返回 origin 远程仓库 URL；如果不存在则返回 None。"""
        if not self.is_repo:
            return None
        cfg = self.refdir / "config"
        remote, url = None, None
        for s in (self._read(cfg) or "").splitlines():
            t = s.strip()
            if t.startswith("[") and t.endswith("]"):
                remote = t.lower()
            elif t.lower().startswith("url =") and remote == '[remote "origin"]':
                url = t.split("=", 1)[1].strip()
                break
        return url


if __name__ == "__main__":
    import time

    g = GitRepo()
    if g.is_repo:
        t0 = time.perf_counter()
        print(f"repo={g.root}\nbranch={g.branch}\ncommit={g.commit}\nmessage={g.message}\norigin={g.origin}")
        dt = (time.perf_counter() - t0) * 1000
        print(f"\n⏱️ Profiling: total {dt:.3f} ms")
