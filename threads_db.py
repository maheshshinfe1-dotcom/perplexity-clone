import json
import sqlite3
import uuid

DB_NAME = "perplexity_threads.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Creates relational tables and B-Tree indexes for saved Perplexity threads."""
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS threads (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            focus TEXT NOT NULL DEFAULT 'web',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            query TEXT NOT NULL,
            focus TEXT NOT NULL DEFAULT 'web',
            sources_json TEXT NOT NULL,
            answer TEXT NOT NULL,
            followups_json TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_turns_thread ON turns(thread_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_threads_updated ON threads(updated_at DESC)")
    conn.commit()
    conn.close()


def create_thread(first_query: str, focus: str = "web") -> dict:
    """Creates a new search thread and returns its metadata."""
    thread_id = uuid.uuid4().hex[:12]
    clean_title = first_query.strip().replace("\n", " ")[:72]
    if len(first_query.strip()) > 72:
        clean_title += "..."

    conn = _connect()
    conn.execute(
        "INSERT INTO threads (id, title, focus) VALUES (?, ?, ?)",
        (thread_id, clean_title, focus),
    )
    conn.commit()
    conn.close()
    return {"id": thread_id, "title": clean_title, "focus": focus}


def thread_exists(thread_id: str) -> bool:
    conn = _connect()
    row = conn.execute("SELECT 1 FROM threads WHERE id = ?", (thread_id,)).fetchone()
    conn.close()
    return row is not None


def save_turn(
    thread_id: str,
    query: str,
    focus: str,
    sources: list[dict],
    answer: str,
    followups: list[str],
) -> None:
    """Persists a completed search turn and refreshes the parent thread timestamp."""
    conn = _connect()
    conn.execute(
        """
        INSERT INTO turns (thread_id, query, focus, sources_json, answer, followups_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (thread_id, query, focus, json.dumps(sources), answer, json.dumps(followups)),
    )
    conn.execute(
        "UPDATE threads SET updated_at = CURRENT_TIMESTAMP, focus = ? WHERE id = ?",
        (focus, thread_id),
    )
    conn.commit()
    conn.close()


def list_threads(limit: int = 30) -> list[dict]:
    """Returns the most recently active threads for the Library sidebar."""
    conn = _connect()
    rows = conn.execute(
        "SELECT id, title, focus, updated_at FROM threads ORDER BY updated_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_thread(thread_id: str) -> dict | None:
    """Loads a thread and all of its saved search turns."""
    conn = _connect()
    thread_row = conn.execute(
        "SELECT id, title, focus, created_at, updated_at FROM threads WHERE id = ?",
        (thread_id,),
    ).fetchone()
    if not thread_row:
        conn.close()
        return None

    turn_rows = conn.execute(
        """
        SELECT id, query, focus, sources_json, answer, followups_json, created_at
        FROM turns
        WHERE thread_id = ?
        ORDER BY id ASC
        """,
        (thread_id,),
    ).fetchall()
    conn.close()

    turns = []
    for row in turn_rows:
        turns.append({
            "id": row["id"],
            "query": row["query"],
            "focus": row["focus"],
            "sources": json.loads(row["sources_json"]),
            "answer": row["answer"],
            "followups": json.loads(row["followups_json"]),
            "created_at": row["created_at"],
        })

    result = dict(thread_row)
    result["turns"] = turns
    return result


def delete_thread(thread_id: str) -> bool:
    """Deletes a thread and cascades removal to all of its saved turns."""
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted