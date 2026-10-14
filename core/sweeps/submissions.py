"""Durable live submissions; Ray Tune remains the only execution scheduler.

The SQLite file belongs on a local disk shared by the submitter and driver.
Use submit(jobs) from Python, or ``python -m core.sweeps.submissions`` for YAML.
"""

import argparse
from contextlib import closing
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3

from omegaconf import OmegaConf

from core.sweeps.jobs import InferenceJob, TrainingJob
from core.sweeps.queue import _training_job


class SubmissionQueue:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()

    def _connect(self):
        # Never silently create a second queue because of a misspelled path.
        return sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=30)

    @classmethod
    def create(cls, path):
        queue = cls(path)
        queue.path.parent.mkdir(parents=True, exist_ok=True)
        with queue.path.open("xb"):
            pass
        with closing(queue._connect()) as db, db:
            db.execute(
                "CREATE TABLE jobs (sequence INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, payload TEXT NOT NULL)"
            )
            db.execute("CREATE TABLE state (closed INTEGER NOT NULL)")
            db.execute("INSERT INTO state VALUES (0)")
        return queue

    def submit(self, jobs):
        """Atomically append a batch. Identical retries are harmless; conflicts fail."""
        payloads = [asdict(_training_job(job)) for job in jobs]
        added = 0
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT closed FROM state").fetchone()[0]:
                raise ValueError("Submission queue is closed")
            for payload in payloads:
                encoded = json.dumps(payload, sort_keys=True)
                existing = db.execute(
                    "SELECT payload FROM jobs WHERE name=?", (payload["name"],)
                ).fetchone()
                if existing:
                    if existing[0] != encoded:
                        raise ValueError(f"Conflicting job name: {payload['name']!r}")
                    continue
                db.execute(
                    "INSERT INTO jobs(name,payload) VALUES (?,?)",
                    (payload["name"], encoded),
                )
                added += 1
        return added

    def read_after(self, sequence):
        with closing(self._connect()) as db, db:
            # Read the open flag and rows in one snapshot, including when closing.
            db.execute("BEGIN")
            opened = not db.execute("SELECT closed FROM state").fetchone()[0]
            rows = db.execute(
                "SELECT sequence,payload FROM jobs WHERE sequence>? ORDER BY sequence",
                (sequence,),
            ).fetchall()
        return [(index, json.loads(payload)) for index, payload in rows], opened

    def close(self):
        """Stop accepting submissions; the scheduler drains accepted work then exits."""
        with closing(self._connect()) as db, db:
            db.execute("UPDATE state SET closed=1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", type=Path)
    parser.add_argument("action", choices=["submit", "status", "close"])
    parser.add_argument(
        "jobs", nargs="?", type=Path, help="YAML list of TrainingJob fields"
    )
    args = parser.parse_args()
    queue = SubmissionQueue(args.queue)
    if args.action == "submit":
        if args.jobs is None:
            parser.error("submit requires a jobs YAML file")
        values = OmegaConf.to_container(OmegaConf.load(args.jobs), resolve=False)
        jobs = [
            TrainingJob(
                **{
                    **value,
                    "inference": tuple(
                        InferenceJob(**x) for x in value.get("inference", [])
                    ),
                }
            )
            for value in values
        ]
        print(f"Accepted {queue.submit(jobs)} new jobs")
    elif args.action == "close":
        queue.close()
        print("Closed submissions; accepted jobs will finish")
    else:
        rows, opened = queue.read_after(0)
        print(
            f"{'Open' if opened else 'Closed'}; {len(rows)} submitted training jobs (execution status is in Ray)"
        )


if __name__ == "__main__":
    main()
