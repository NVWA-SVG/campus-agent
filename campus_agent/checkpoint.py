"""受限保留、可按回合回滚的 SQLite LangGraph checkpoint。"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Iterator

from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.sqlite import SqliteSaver


class PrunableSqliteSaver(SqliteSaver):
    """为线性会话提供原子回合回滚和有界物理保留。

    Campus Agent 在 Web 层为同一 session 加锁，因此每个 thread 都是一条线性链，
    不使用 LangGraph time travel/fork。这个约束允许成功后只保留最新快照。
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        serde: SerializerProtocol | None = None,
        max_threads: int = 256,
    ) -> None:
        if max_threads < 1:
            raise ValueError("max_threads 必须至少为 1")
        super().__init__(conn, serde=serde)
        self.max_threads = max_threads

    def latest_id(self, thread_id: str, checkpoint_ns: str = "") -> str | None:
        with self.cursor(transaction=False) as cursor:
            row = cursor.execute(
                """
                SELECT checkpoint_id
                FROM checkpoints
                WHERE thread_id = ? AND checkpoint_ns = ?
                ORDER BY checkpoint_id DESC
                LIMIT 1
                """,
                (str(thread_id), checkpoint_ns),
            ).fetchone()
        return str(row[0]) if row is not None else None

    def rollback_after(
        self,
        thread_id: str,
        keep_id: str | None,
        checkpoint_ns: str = "",
    ) -> None:
        """删除本轮产生的后代；链不连续时拒绝猜测性删除。"""

        with self._transaction() as cursor:
            rows = cursor.execute(
                """
                SELECT checkpoint_id, parent_checkpoint_id
                FROM checkpoints
                WHERE thread_id = ? AND checkpoint_ns = ?
                """,
                (str(thread_id), checkpoint_ns),
            ).fetchall()
            if not rows:
                return
            parents = {
                str(checkpoint_id): (
                    str(parent_checkpoint_id)
                    if parent_checkpoint_id is not None
                    else None
                )
                for checkpoint_id, parent_checkpoint_id in rows
            }
            latest_id = max(parents)
            if latest_id == keep_id:
                return
            if keep_id is None:
                remove_ids = tuple(parents)
            else:
                remove: list[str] = []
                cursor_id: str | None = latest_id
                while cursor_id is not None and cursor_id != keep_id:
                    remove.append(cursor_id)
                    cursor_id = parents.get(cursor_id)
                if cursor_id != keep_id:
                    raise RuntimeError("checkpoint 链与回合起点不一致，已拒绝回滚")
                remove_ids = tuple(remove)
            self._delete_ids(cursor, thread_id, checkpoint_ns, remove_ids)

    def prune_thread(
        self,
        thread_id: str,
        *,
        keep: int = 1,
        checkpoint_ns: str = "",
    ) -> None:
        if keep < 1:
            raise ValueError("keep 必须至少为 1")
        with self._transaction() as cursor:
            rows = cursor.execute(
                """
                SELECT checkpoint_id
                FROM checkpoints
                WHERE thread_id = ? AND checkpoint_ns = ?
                ORDER BY checkpoint_id DESC
                """,
                (str(thread_id), checkpoint_ns),
            ).fetchall()
            retained = tuple(str(row[0]) for row in rows[:keep])
            removed = tuple(str(row[0]) for row in rows[keep:])
            self._delete_ids(cursor, thread_id, checkpoint_ns, removed)
            if retained:
                # 被保留链的最老节点不再指向已经物理删除的祖先。
                cursor.execute(
                    """
                    UPDATE checkpoints
                    SET parent_checkpoint_id = NULL
                    WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                    """,
                    (str(thread_id), checkpoint_ns, retained[-1]),
                )

    def startup_prune(self, *, keep_per_thread: int = 1) -> None:
        """启动时压缩历史版本，并限制最多保留最近的会话数量。"""

        with self.cursor(transaction=False) as cursor:
            threads = [
                (str(thread_id), str(checkpoint_ns), str(latest_id))
                for thread_id, checkpoint_ns, latest_id in cursor.execute(
                    """
                    SELECT thread_id, checkpoint_ns, MAX(checkpoint_id)
                    FROM checkpoints
                    GROUP BY thread_id, checkpoint_ns
                    ORDER BY MAX(checkpoint_id) DESC
                    """
                ).fetchall()
            ]
        for thread_id, checkpoint_ns, _ in threads[: self.max_threads]:
            self.prune_thread(
                thread_id,
                keep=keep_per_thread,
                checkpoint_ns=checkpoint_ns,
            )
        for thread_id, checkpoint_ns, _ in threads[self.max_threads :]:
            self._delete_namespace(thread_id, checkpoint_ns)

    def truncate_wal(self) -> bool:
        """在没有活跃 Graph 时回收 WAL；返回是否完成（而非 busy）。"""

        with self.lock:
            self.setup()
            row = self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        return row is not None and int(row[0]) == 0

    def enforce_thread_limit(self) -> int:
        """用单个事务淘汰超过上限的旧 session，返回淘汰数量。"""

        with self._transaction() as cursor:
            expired = [
                (str(thread_id), str(checkpoint_ns))
                for thread_id, checkpoint_ns in cursor.execute(
                    """
                    SELECT thread_id, checkpoint_ns
                    FROM checkpoints
                    GROUP BY thread_id, checkpoint_ns
                    ORDER BY MAX(checkpoint_id) DESC
                    LIMIT -1 OFFSET ?
                    """,
                    (self.max_threads,),
                ).fetchall()
            ]
            cursor.executemany(
                "DELETE FROM writes WHERE thread_id = ? AND checkpoint_ns = ?",
                expired,
            )
            cursor.executemany(
                "DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ?",
                expired,
            )
        return len(expired)

    def _delete_namespace(self, thread_id: str, checkpoint_ns: str) -> None:
        with self._transaction() as cursor:
            cursor.execute(
                "DELETE FROM writes WHERE thread_id = ? AND checkpoint_ns = ?",
                (thread_id, checkpoint_ns),
            )
            cursor.execute(
                "DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ?",
                (thread_id, checkpoint_ns),
            )

    @staticmethod
    def _delete_ids(
        cursor: sqlite3.Cursor,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_ids: Iterable[str],
    ) -> None:
        values = tuple(
            (str(thread_id), checkpoint_ns, str(checkpoint_id))
            for checkpoint_id in checkpoint_ids
        )
        if not values:
            return
        cursor.executemany(
            """
            DELETE FROM writes
            WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
            """,
            values,
        )
        cursor.executemany(
            """
            DELETE FROM checkpoints
            WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
            """,
            values,
        )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Cursor]:
        """让多条清理 SQL 真正做到全部成功或全部回滚。"""

        with self.lock:
            self.setup()
            cursor = self.conn.cursor()
            try:
                cursor.execute("BEGIN IMMEDIATE")
                yield cursor
            except BaseException:
                self.conn.rollback()
                raise
            else:
                self.conn.commit()
            finally:
                cursor.close()
