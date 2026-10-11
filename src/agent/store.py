"""Small SQLite checkpoint store used to resume approval-gated runs."""
import json
from pathlib import Path
import sqlite3
import threading

from .models import AgentRun


class AgentRunStore:
    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS agent_runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    state_json TEXT NOT NULL
                )"""
            )

    def save(self, run: AgentRun) -> AgentRun:
        payload = run.model_dump_json()
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO agent_runs(run_id, status, updated_at, state_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    state_json=excluded.state_json""",
                (run.run_id, run.status, run.updated_at, payload),
            )
        return run

    def get(self, run_id: str) -> AgentRun | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM agent_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return AgentRun.model_validate(json.loads(row["state_json"])) if row else None
