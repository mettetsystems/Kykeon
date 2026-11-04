import os
from pathlib import Path

import pytest

from kykeon_backend.errors import PathRejected
from kykeon_backend.paths import contained, contained_any


def test_rejects_traversal_and_missing_roots(tmp_path: Path):
    root = tmp_path / "models"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(PathRejected, match="escapes"):
        contained(root, root / ".." / "outside")
    with pytest.raises(PathRejected, match="No model roots"):
        contained_any((), root)


def test_rejects_symlink_that_leaves_the_root(tmp_path: Path):
    root = tmp_path / "models"
    root.mkdir()
    outside = tmp_path / "secret"
    outside.mkdir()
    (outside / "checkpoint").mkdir()
    link = root / "alias"
    link.symlink_to(outside / "checkpoint", target_is_directory=True)
    with pytest.raises(PathRejected, match="escapes"):
        contained(root, link)


def test_accepts_a_real_directory_inside_the_root(tmp_path: Path):
    root = tmp_path / "models"
    nested = root / "llama"
    nested.mkdir(parents=True)
    assert contained(root, nested) == nested.resolve()


def test_rejects_null_bytes(tmp_path: Path):
    root = tmp_path / "models"
    root.mkdir()
    with pytest.raises(PathRejected, match="null"):
        contained(root, Path("/tmp/nope\x00"))


def test_environment_does_not_execute_custom_source(tmp_path: Path):
    sentinel = tmp_path / "pwned"
    source = "import os; os.system('touch " + str(sentinel) + "')"
    assert "os.system" in source
    assert not sentinel.exists()
    os.environ.pop("KYKEON_EXECUTED_CUSTOM", None)
