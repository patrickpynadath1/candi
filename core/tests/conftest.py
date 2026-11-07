"""Shared test isolation."""

import os

import pytest

# Under `uv run`, Ray otherwise builds a runtime env that uploads the current
# directory (with any large untracked files) to each local test cluster. It must
# be set before Ray is first imported, so set it here rather than in a test body.
os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"


@pytest.fixture(autouse=True)
def isolate_from_enclosing_ray_job(monkeypatch):
    # When the suite itself runs as a Ray Job (e.g. on the GPU server), Ray
    # exports the job's address and runtime env. Tests start their own local
    # clusters; inheriting the outer job's working_dir package breaks them,
    # and attaching to the outer cluster would touch live work.
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("RAY_JOB_CONFIG_JSON_ENV_VAR", raising=False)
