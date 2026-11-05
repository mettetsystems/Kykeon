from __future__ import annotations

from pathlib import Path

from .errors import PathRejected


def _reject(message: str) -> None:
    raise PathRejected(message)


def contained(root: Path, candidate: Path) -> Path:
    """Resolve `candidate` and require the real path to stay inside `root`.

    Symlinks are followed. A link that lands outside the root is rejected,
    including intermediate links and `..` traversal.
    """
    if "\x00" in str(candidate) or "\x00" in str(root):
        _reject("Paths cannot contain null bytes.")
    try:
        root_real = root.expanduser().resolve(strict=True)
    except FileNotFoundError:
        _reject(f"Configured root does not exist: {root}")
    if not root_real.is_dir():
        _reject(f"Configured root is not a directory: {root}")
    raw = candidate.expanduser()
    if not raw.is_absolute():
        raw = root_real / raw
    try:
        real = raw.resolve(strict=True)
    except FileNotFoundError:
        _reject(f"Path does not exist: {candidate}")
    try:
        real.relative_to(root_real)
    except ValueError:
        _reject(f"Path escapes the configured root: {candidate}")
    return real


def within_any(roots: list[Path] | tuple[Path, ...], candidate: Path) -> Path:
    """Resolve a file or directory and require its real path to stay inside a root.

    This follows symlinks. A link whose target leaves every configured root is
    rejected before the caller reads the target.
    """
    if not roots:
        _reject("No configured roots are available for this path.")
    if "\x00" in str(candidate):
        _reject("Paths cannot contain null bytes.")
    try:
        real = candidate.expanduser().resolve(strict=True)
    except FileNotFoundError:
        _reject(f"Path does not exist: {candidate}")
    except OSError as exc:
        _reject(f"Path cannot be resolved: {candidate} ({exc})")
    errors: list[str] = []
    for root in roots:
        try:
            root_real = root.expanduser().resolve(strict=True)
        except FileNotFoundError:
            errors.append(f"Configured root does not exist: {root}")
            continue
        try:
            real.relative_to(root_real)
            return real
        except ValueError:
            errors.append(f"{candidate} resolves outside {root_real}")
    _reject(
        "Path escapes the configured roots. "
        + " ".join(errors[-3:])
    )
    raise AssertionError("unreachable")


def safe_rmtree(root: Path, target: Path) -> None:
    """Remove a job or artifact path only when its real location stays under root."""
    import shutil

    if not target.exists() and not target.is_symlink():
        return
    if target.is_symlink():
        real = target.resolve(strict=False)
        try:
            real.relative_to(root.expanduser().resolve(strict=True))
        except ValueError:
            _reject(f"Refusing to delete a symlink that leaves the artifact root: {target}")
        target.unlink()
        return
    within_any((root,), target)
    shutil.rmtree(target)


def contained_any(roots: list[Path] | tuple[Path, ...], candidate: Path) -> Path:
    if not roots:
        _reject(
            "No model roots are configured. Set KYKEON_MODEL_ROOTS to the "
            "directories that may be registered."
        )
    errors: list[str] = []
    for root in roots:
        try:
            return contained(root, candidate)
        except PathRejected as exc:
            errors.append(exc.message)
    _reject(
        "Path is outside every configured model root. "
        + " ".join(errors[-3:])
    )
    raise AssertionError("unreachable")


def output_path(root: Path, target: Path) -> Path:
    """Confine outputs; reject symlinks in every component below the root.

    The artifact root is a user-owned directory, not a shared untrusted tree.
    Missing components are checked lexically before they are created.
    """
    root = root.expanduser().absolute()
    target = target.expanduser().absolute()
    if root.is_symlink():
        _reject(f"Output root must not be a symlink: {root}")
    try:
        relative = target.relative_to(root)
    except ValueError:
        _reject(f"Output escapes the artifact root: {target}")
    if ".." in relative.parts:
        _reject(f"Output traversal is not allowed: {target}")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            _reject(f"Output symlinks are not allowed: {current}")
    return target


def atomic_json(root: Path, target: Path, value) -> None:
    import json
    import os
    import tempfile

    target = output_path(root, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".kykeon-", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        output_path(root, target)
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)
