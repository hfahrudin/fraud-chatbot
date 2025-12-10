import sqlite3
from typing import List, Dict, Any, Optional
import logging
from utils import create_tab_db
logger = logging.getLogger(__name__)

class SQLiteDBManager:
    """
    Manages SQLite database interactions for querying schema and data.
    """

    def __init__(self, db_path: str):
        """
        Initializes the SQLDBManager with the path to the SQLite database.

        Args:
            db_path: Path to the SQLite database file.
        """
        create_tab_db()
        self.db_path = db_path
        logger.debug(f"SQLDBManager initialized with database: {self.db_path}")

    def _get_db_connection(self) -> sqlite3.Connection:
        """
        Establishes and returns a connection to the SQLite database.
        Rows are returned as sqlite3.Row objects (dict-like).
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dict-like objects
        return conn

    def get_column_names(self, table_name: str) -> List[str]:
        """
        Retrieves all column names for a given table.

        Args:
            table_name: The name of the table.

        Returns:
            A list of column names.
            Returns an empty list if the table does not exist or an error occurs.
        """
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns_info = cursor.fetchall()
            return [col['name'] for col in columns_info]
        except sqlite3.Error as e:
            logger.error(f"Error getting column names for table '{table_name}': {e}")
            return []
        finally:
            if conn:
                conn.close()

    def execute_read_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Executes a read-only SQL query and returns the results.

        Args:
            query: The SQL query string to execute.

        Returns:
            A list of dictionaries, where each dictionary represents a row
            and keys are column names.
            Returns an empty list if no results or an error occurs.
        """
        conn: Optional[sqlite3.Connection] = None
        results: List[Dict[str, Any]] = []
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()

            # Ensure the query is read-only (basic check)
            # This is a very basic check and might not catch all write operations disguised
            # as reads (e.g., UDFs that write). For production, consider a more robust
            # SQL parser or restricting user permissions.
            if not query.strip().upper().startswith("SELECT"):
                logger.warning(f"Attempted to execute non-SELECT query: {query}")
                raise ValueError("Only SELECT queries are allowed for read operations.")

            cursor.execute(query)
            rows = cursor.fetchall()

            for row in rows:
                results.append(dict(row))
            return results
        except sqlite3.Error as e:
            logger.error(f"Error executing query '{query}': {e}")
            return []
        finally:
            if conn:
                conn.close()
