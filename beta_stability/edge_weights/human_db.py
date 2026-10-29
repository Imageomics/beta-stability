import logging
import sqlite3
import os
from datetime import datetime

logger = logging.getLogger("beta_stability")

class human_db(object):
    def __init__(self, db_path, data_df, node2uuid, id_key='uuid', max_reviews=None):
        self.db_path = db_path
        # At most this many pairs of this run are ever answered or waiting in
        # the UI; None leaves the number unbounded.
        self.max_reviews = max_reviews
        self.data_df = data_df
        self.node2uuid = node2uuid
        self.uuid2node = {val: key for (key, val) in node2uuid.items()}
        self.id_key = id_key
        self.path_key = "file_path" if "file_path" in self.data_df.columns else "image_path"

    # --- internal helpers ---
    @staticmethod
    def _column_exists(conn, table, column):
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        cols = [row[1] for row in cur.fetchall()]
        return column in cols

    def init_db(self, db_path="./zebra_verification.db"):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS image_verification (
            id TEXT PRIMARY KEY,
            uuid1 TEXT,
            image1_path TEXT,
            bbox1 TEXT,
            cluster1 TEXT,
            uuid2 TEXT,
            image2_path TEXT,
            bbox2 TEXT,
            cluster2 TEXT,
            score REAL,
            status TEXT CHECK(status IN ('awaiting', 'in_progress', 'checked', 'sent')) DEFAULT 'awaiting',
            decision TEXT CHECK(decision IN ('none', 'correct', 'incorrect', 'cant_tell')) DEFAULT 'none',
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            instance_id TEXT,
            heartbeat TIMESTAMP,
            read_at TIMESTAMP
        )
        """)
        conn.commit()

        # Safe migration if table already existed without score column
        if not self._column_exists(conn, "image_verification", "score"):
            cursor.execute("ALTER TABLE image_verification ADD COLUMN score REAL NOT NULL DEFAULT 0.0")
            conn.commit()

        # __call__ stamps read_at when it consumes decisions; a table created by an
        # older version of this file lacks the column and the UPDATE would fail.
        if not self._column_exists(conn, "image_verification", "read_at"):
            cursor.execute("ALTER TABLE image_verification ADD COLUMN read_at TIMESTAMP")
            conn.commit()

        conn.close()

    def get_image_path(self, uuid):
        file_name = self.data_df.loc[self.data_df[self.id_key] == uuid, self.path_key].squeeze()
        return str(file_name)

    def get_bbox(self, uuid):
        import json
        import pandas as pd
        import numpy as np
        if 'bbox' in self.data_df.columns:
            bbox = self.data_df.loc[self.data_df[self.id_key] == uuid, 'bbox'].squeeze()
            if bbox is None:
                return None
            if isinstance(bbox, float) and pd.isna(bbox):
                return None
            if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) >= 4:
                x, y, w, h = bbox[:4]
                return json.dumps([int(x), int(y), int(x + w), int(y + h)])
            return None
        return None

    def get_cluster(self, uuid):
        if 'tracking_id' in self.data_df.columns:
            cluster = self.data_df.loc[self.data_df[self.id_key] == uuid, 'tracking_id'].squeeze()
            return str(cluster) if cluster is not None and str(cluster) != 'nan' else ""
        return ""

    def convert_query(self, n0, n1, score):
        uuid1 = self.node2uuid[n0]
        uuid2 = self.node2uuid[n1]
        unique_id = f"{uuid1}____{uuid2}" if uuid1 < uuid2 else f"{uuid2}____{uuid1}"
        return (
            unique_id,
            uuid1, self.get_image_path(uuid1), self.get_bbox(uuid1), self.get_cluster(uuid1),
            uuid2, self.get_image_path(uuid2), self.get_bbox(uuid2), self.get_cluster(uuid2),
            float(score)
        )

    def _within_budget(self, cursor, db_query):
        """Drop new pairs that would take the UI past max_reviews.

        Every pair of this run already in the table, answered or still
        waiting, holds a slot; pairs already in the table stay in the query so
        their answers are still read back.
        """
        cursor.execute("SELECT id, uuid1, uuid2 FROM image_verification")
        known = {pair_id for (pair_id, uuid1, uuid2) in cursor.fetchall()
                 if uuid1 in self.uuid2node and uuid2 in self.uuid2node}
        slots = max(0, self.max_reviews - len(known))
        new = [row for row in db_query if row[0] not in known]
        if len(new) > slots:
            logger.info(f"Review budget: {len(known)} of {self.max_reviews} pairs already "
                        f"in the database, dropping {len(new) - slots} new pairs")
        keep = {row[0] for row in new[:slots]}
        return [row for row in db_query if row[0] in known or row[0] in keep]

    def withdraw_pending(self):
        """Remove the pairs nobody has answered, once the run no longer needs them.

        Without this the UI keeps offering the last requested batch after the
        run has finished, and those answers would never be read.
        """
        if not os.path.exists(self.db_path):
            return 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM image_verification
                WHERE status IN ('awaiting', 'in_progress') AND decision = 'none'
            """)
            withdrawn = cursor.rowcount
        logger.info(f"Withdrew {withdrawn} unanswered pairs from the review database")
        return withdrawn

    def __call__(self, query):
        """
        query: iterable of (n0, n1, score)
        returns: (reviews, False) where reviews is [(node0, node1, is_correct), ...]
        """
        logger.info(f"__call__ invoked with {len(list(query)) if hasattr(query, '__len__') else 'unknown number of'} query pairs")
        query = list(query)  # Convert to list to allow multiple iterations
        logger.debug(f"Query contains {len(query)} pairs")

        if not os.path.exists(self.db_path):
            logger.warning(f"Database does not exist at {self.db_path}")
            print("DB doesn't exist")
            logger.info(f"Initializing database at {self.db_path}")
            self.init_db(self.db_path)
            if not os.path.exists(self.db_path):
                logger.error(f"Failed to create database at {self.db_path}")
                raise RuntimeError(f"Failed to create DB at {self.db_path}")
            logger.info(f"Database successfully created at {self.db_path}")
        else:
            logger.debug(f"Database exists at {self.db_path}")

        reviews = []
        logger.info(f"Connecting to database at {self.db_path}")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Prepare inserts (always with score)
            logger.debug("Converting query pairs to database format")
            db_query = [self.convert_query(n0, n1, score) for (n0, n1, score) in query]
            logger.info(f"Converted {len(db_query)} query pairs")

            if db_query and self.max_reviews is not None:
                db_query = self._within_budget(cursor, db_query)

            if db_query:
                logger.debug(f"Inserting {len(db_query)} pairs into image_verification table")
                cursor.executemany("""
                    INSERT OR IGNORE INTO image_verification
                    (id, uuid1, image1_path, bbox1, cluster1,
                     uuid2, image2_path, bbox2, cluster2, score, status, decision)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'awaiting', 'none')
                """, db_query)

            inserted_count = cursor.rowcount if db_query else 0
            logger.info(f"Inserted {inserted_count} new pairs for checking")
            print(f"Sent {inserted_count} new pairs for checking.")

            # Fetch checked decisions
            logger.debug("Fetching checked decisions from database")
            cursor.execute("""
                SELECT id, uuid1, uuid2, decision FROM image_verification
                WHERE status = 'checked'
            """)
            checked_results = cursor.fetchall()
            logger.info(f"Found {len(checked_results)} checked results")

            checked_results = [(r1, uuid1, uuid2, decision) for (r1, uuid1, uuid2, decision) in checked_results if uuid1 in self.uuid2node and uuid2 in self.uuid2node]
            logger.debug(f"Filtered to {len(checked_results)} checked results with valid UUIDs")

            # Check if any query pairs are already 'sent'
            sent_results = []
            if db_query:
                query_ids = [row[0] for row in db_query]
                logger.debug(f"Checking for {len(query_ids)} query pairs that are already sent")
                placeholders = ','.join('?' * len(query_ids))
                cursor.execute(f"""
                    SELECT id, uuid1, uuid2, decision FROM image_verification
                    WHERE status = 'sent' AND id IN ({placeholders})
                """, query_ids)
                sent_results = cursor.fetchall()
                logger.info(f"Found {len(sent_results)} results already marked as sent")

                sent_results = [(r1, uuid1, uuid2, decision) for (r1, uuid1, uuid2, decision) in sent_results if uuid1 in self.uuid2node and uuid2 in self.uuid2node]
                logger.debug(f"Filtered to {len(sent_results)} sent results with valid UUIDs")

                # Stamp read_at for re-read sent results
                if sent_results:
                    now_sent = datetime.now().isoformat()
                    cursor.executemany(
                        "UPDATE image_verification SET read_at=? WHERE id=?",
                        [(now_sent, res[0]) for res in sent_results]
                    )

            # Update checked to sent and stamp read_at
            now = datetime.now().isoformat()
            if checked_results:
                logger.info(f"Updating {len(checked_results)} checked results to sent status")
                cursor.executemany(
                    "UPDATE image_verification SET status='sent', read_at=? WHERE id=?",
                    [(now, res[0]) for res in checked_results]
                )
                logger.debug(f"Successfully updated status for {len(checked_results)} results")

            # Merge results
            result = checked_results + sent_results
            logger.info(f"Total merged results: {len(result)} ({len(checked_results)} checked + {len(sent_results)} sent)")

            # "Can't tell" carries no verdict (None) rather than reading as "different".
            verdict = {'correct': True, 'incorrect': False}
            reviews = [
                (self.uuid2node[uuid1], self.uuid2node[uuid2], verdict.get(decision))
                for (_, uuid1, uuid2, decision) in result
            ]
            logger.info(f"Prepared {len(reviews)} reviews for return")
            logger.debug(f"Reviews breakdown: {sum(1 for r in reviews if r[2])} correct, {sum(1 for r in reviews if not r[2])} incorrect")

        logger.info(f"__call__ completed, returning {len(reviews)} reviews")
        return reviews, False
