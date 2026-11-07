"""Caches are published whole, so a concurrent reader never sees a partial one."""

from core.data.cache import save_dataset


class FakeDataset:
    def __init__(self, marker):
        self.marker = marker

    def save_to_disk(self, path):
        from pathlib import Path

        Path(path).mkdir(parents=True)
        (Path(path) / "data").write_text(self.marker)


def test_publishes_by_rename_and_leaves_no_temporary(tmp_path):
    cache = tmp_path / "nested" / "sudoku_cache"
    save_dataset(FakeDataset("first"), cache)
    assert (cache / "data").read_text() == "first"
    assert [p.name for p in cache.parent.iterdir()] == ["sudoku_cache"]


def test_keeps_a_cache_another_process_published_first(tmp_path):
    cache = tmp_path / "sudoku_cache"
    save_dataset(FakeDataset("first"), cache)
    save_dataset(FakeDataset("second"), cache)
    assert (cache / "data").read_text() == "first"
    assert [p.name for p in tmp_path.iterdir()] == ["sudoku_cache"]
