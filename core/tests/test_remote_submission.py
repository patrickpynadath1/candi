"""Laptop submissions upload a bounded, content-identified snapshot to the server."""

from pathlib import Path
import subprocess

import pytest

from scripts.cluster import sddc_remote as remote


@pytest.mark.parametrize(
    "path, excluded",
    [
        ("core/outputs/run/config.yaml", True),
        ("core/data_cache/sudoku/train.npy", True),
        ("core/models/modular.py", False),
        ("configs/sweep/common.yaml", False),
        ("nested/model.ckpt", True),
        ("wandb/run-1/log.txt", True),
        (".env.local", True),
        ("core/outputs_helper.py", False),
    ],
)
def test_excludes_follow_gitignore_semantics(path, excluded):
    assert remote._excluded(path) is excluded


@pytest.fixture
def repo(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    (tmp_path / ".gitignore").write_text("ignored.txt\n")
    (tmp_path / "uv.lock").write_text("lock\n")
    (tmp_path / "core/outputs").mkdir(parents=True)
    (tmp_path / "core/outputs/big.bin").write_bytes(b"x")
    (tmp_path / "main.py").write_text("print(1)\n")
    git("add", ".gitignore", "uv.lock", "main.py")
    git("commit", "-qm", "init")
    return tmp_path


def test_snapshot_files_and_identity(repo):
    (repo / "ignored.txt").write_text("secret\n")
    (repo / "new.py").write_text("x = 1\n")
    assert remote.upload_files(repo) == [".gitignore", "main.py", "new.py", "uv.lock"]

    revision, content, lock = remote.source_identity(repo)
    assert len(revision) == 40 and len(content) == len(lock) == 12
    # Identical content reuses the snapshot; any uploaded change gets a new one.
    assert remote.source_identity(repo) == (revision, content, lock)
    (repo / "ignored.txt").write_text("changed\n")
    assert remote.source_identity(repo)[1] == content
    (repo / "new.py").write_text("x = 2\n")
    assert remote.source_identity(repo)[1] != content


def test_upload_size_is_bounded(repo, monkeypatch):
    monkeypatch.setattr(remote, "MAX_UPLOAD_BYTES", 4)
    with pytest.raises(SystemExit, match="Upload would be"):
        remote.source_identity(repo)


def test_runtime_env_targets_persistent_storage():
    env = remote.runtime_env("a" * 40, "b" * 12, "c" * 12, storage_root="/srv/sddc")
    assert env["py_executable"] == "/srv/sddc/envs/cccccccccccc/bin/python"
    assert env["env_vars"]["SDDC_STORAGE_ROOT"] == "/srv/sddc"
    assert env["env_vars"]["SDDC_SOURCE_ID"] == "aaaaaaaaaaaa-bbbbbbbbbbbb"
    assert "core/data_cache" in env["excludes"]
    assert Path(env["working_dir"]) == remote.REPO
