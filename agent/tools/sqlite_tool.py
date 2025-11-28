# agent/tools/sqlite_tool.py

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SQLResult:
    sql: str
    columns: List[str]
    rows: List[Tuple[Any, ...]]
    error: Optional[str] = None


@dataclass
class SQLiteTool:
    db_path: str
    _conn: sqlite3.Connection = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def get_tables(self) -> List[str]:
        cur = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view');"
        )
        return [row["name"] for row in cur.fetchall()]

    def get_schema_str(self) -> str:
        """Return a human-readable schema description via PRAGMA."""
        tables = self.get_tables()
        parts: List[str] = []
        for t in tables:
            parts.append(f"TABLE {t}")
            try:
                cur = self._conn.execute(f"PRAGMA table_info('{t}')")
                cols = [f"{r['name']} {r['type']}" for r in cur.fetchall()]
                parts.append("  columns: " + ", ".join(cols))
            except Exception as e:
                parts.append(f"  <error reading schema: {e}>")
        return "\n".join(parts)

    def run_sql(self, sql: str) -> SQLResult:
        sql = sql.strip().rstrip(";") + ";"  # normalize a bit
        try:
            cur = self._conn.execute(sql)
            rows = cur.fetchall()
            columns = [c[0] for c in cur.description] if cur.description else []
            return SQLResult(sql=sql, columns=columns, rows=[tuple(r) for r in rows])
        except Exception as e:
            return SQLResult(sql=sql, columns=[], rows=[], error=str(e))
