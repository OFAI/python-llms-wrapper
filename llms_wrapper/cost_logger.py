"""
Simple cost logger to get used optionally by LLMS/LLM
"""

import sqlite3
import os
import calendar
from typing import Any, Optional
from pathlib import Path
import json
import socket
import getpass
import datetime
import argparse


def get_username() -> str:
    """
    Get the current username in a portable way.

    Returns:
        Username as a string
    """
    try:
        # getpass.getuser() works on Windows, Linux, and macOS
        # It checks environment variables in order: LOGNAME, USER, LNAME, USERNAME
        return getpass.getuser()
    except Exception:
        # Fallback to environment variables
        return os.environ.get('USER') or os.environ.get('USERNAME') or 'unknown'


def get_hostname() -> str:
    """
    Get the current hostname in a portable way.

    Returns:
        Hostname as a string
    """
    try:
        # socket.gethostname() works on Windows, Linux, and macOS
        return socket.gethostname()
    except Exception:
        return 'unknown'


class Log2Sqlite:
    """SQLite logger for API usage tracking with aggregation support."""

    SCHEMA = {
        'model': 'TEXT',
        'modelalias': 'TEXT',
        'hostname': 'TEXT',
        'user': 'TEXT',
        'project': 'TEXT',
        'task': 'TEXT',
        'note': 'TEXT',
        'datetime': 'TEXT',
        'cost': 'REAL',
        'input_tokens': 'INTEGER',
        'output_tokens': 'INTEGER',
        'apikey_name': 'TEXT'
    }

    def __init__(self, db_path: str, **defaults):
        """
        Initialize the logger with a database path and default field values.

        Args:
            db_path: Path to the SQLite database file
            **defaults: Default values for any fields (e.g., project='myproject')

        Raises:
            Exception: If database initialization fails
        """
        db_path = os.path.expanduser(db_path)
        db_path = os.path.expandvars(db_path)
        self.db_path = Path(db_path)

        if defaults.get("user") is None:
            defaults["user"] = get_username()
        if defaults.get("hostname") is None:
            defaults["hostname"] = get_hostname()


        self.defaults = defaults

        # Validate that defaults only contain known fields
        invalid_fields = set(defaults.keys()) - set(self.SCHEMA.keys())
        if invalid_fields:
            raise Exception(f"Invalid default fields: {invalid_fields}")
        if os.path.exists(self.db_path):
            return
        try:
            self._initialize_database()
        except Exception as e:
            raise Exception(f"Failed to initialize database: {e}") from e

    def _initialize_database(self):
        """Create the table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Enable WAL mode for better concurrency
            conn.execute('PRAGMA journal_mode=WAL')

            # Create table
            fields = ', '.join([f'{name} {dtype}' for name, dtype in self.SCHEMA.items()])
            conn.execute(f'''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    {fields}
                )
            ''')

            # Create indexes for commonly filtered fields
            # These are likely to be used in WHERE clauses for aggregation
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_project 
                ON logs(project)
            ''')

            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_user 
                ON logs(user)
            ''')

            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_model 
                ON logs(model)
            ''')

            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_modelalias 
                ON logs(modelalias)
            ''')

            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_datetime 
                ON logs(datetime)
            ''')

            # Composite index for common query patterns
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_project_user_datetime 
                ON logs(project, user, datetime)
            ''')

            conn.commit()
        finally:
            conn.close()

    def log(self, row: dict):
        """
        Log a row to the database.

        Args:
            row: Dictionary of field/value pairs

        Raises:
            Exception: If logging fails or invalid fields are provided
        """
        # Validate fields
        invalid_fields = set(row.keys()) - set(self.SCHEMA.keys())
        if invalid_fields:
            raise Exception(f"Invalid fields in row: {invalid_fields}")

        # the datetime is always set fixed here!
        row["datetime"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


        # Merge defaults with provided row (row takes precedence)
        merged_row = {**self.defaults, **row}

        # Prepare insert statement
        fields = list(self.SCHEMA.keys())
        placeholders = ', '.join(['?'] * len(fields))
        field_names = ', '.join(fields)

        # Get values in the correct order, None for missing fields
        values = [merged_row.get(field) for field in fields]

        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            try:
                conn.execute(
                    f'INSERT INTO logs ({field_names}) VALUES ({placeholders})',
                    values
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            raise Exception(f"Failed to log row: {e}") from e

    def _build_where_clause(self) -> tuple[str, list]:
        """Build WHERE clause from defaults for aggregation queries."""
        if not self.defaults:
            return '', []

        conditions = []
        values = []

        for field, value in self.defaults.items():
            if value is not None:
                conditions.append(f'{field} = ?')
                values.append(value)

        if conditions:
            where = 'WHERE ' + ' AND '.join(conditions)
            return where, values
        return '', []

    def export(self, file: str):
        """
        Export all rows from the logs table to a JSONL file.

        Args:
            file: Path to the output JSONL file

        Raises:
            Exception: If export fails
        """
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            try:
                # Get all rows
                conn.row_factory = sqlite3.Row  # Access columns by name
                cursor = conn.execute('SELECT * FROM logs ORDER BY id')

                with open(file, 'w', encoding='utf-8') as f:
                    for row in cursor:
                        # Convert Row object to dict, excluding the id field
                        row_dict = {key: row[key] for key in row.keys() if key != 'id'}
                        f.write(json.dumps(row_dict) + '\n')
            finally:
                conn.close()
        except Exception as e:
            raise Exception(f"Failed to export to {file}: {e}") from e

    def import_file(self, file: str):
        """
        Import rows from a JSONL file and add them to the logs table.
        Does not apply default values - imports data exactly as provided.

        Args:
            file: Path to the input JSONL file

        Raises:
            Exception: If import fails or file contains invalid data
        """
        try:
            with open(file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue

                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise Exception(f"Invalid JSON on line {line_num}: {e}") from e

                    if not isinstance(row, dict):
                        raise Exception(f"Line {line_num} is not a JSON object")

                    # Validate fields
                    invalid_fields = set(row.keys()) - set(self.SCHEMA.keys())
                    if invalid_fields:
                        raise Exception(f"Invalid fields on line {line_num}: {invalid_fields}")

                    # Prepare insert statement
                    fields = list(self.SCHEMA.keys())
                    placeholders = ', '.join(['?'] * len(fields))
                    field_names = ', '.join(fields)

                    # Get values in the correct order, None for missing fields
                    # Do NOT merge with defaults
                    values = [row.get(field) for field in fields]

                    # Insert directly without applying defaults
                    conn = sqlite3.connect(self.db_path, timeout=5.0)
                    try:
                        conn.execute(
                            f'INSERT INTO logs ({field_names}) VALUES ({placeholders})',
                            values
                        )
                        conn.commit()
                    finally:
                        conn.close()

        except Exception as e:
            if "Failed to import from" not in str(e):
                raise Exception(f"Failed to import from {file}: {e}") from e
            raise

    def rows(self, model=None, modelalias=None, hostname=None, user=None,
             project=None, task=None, note=None, apikey_name=None,
             date_from=None, date_to=None) -> list[dict]:
        """
        Get all rows matching specified criteria.

        Args:
            model: Filter by model name
            modelalias: Filter by model alias
            hostname: Filter by hostname
            user: Filter by user
            project: Filter by project
            task: Filter by task
            note: Filter by note
            apikey_name: Filter by API key name
            date_from: Filter by datetime >= this value (inclusive)
            date_to: Filter by datetime <= this value (inclusive)

        Returns:
            List of dictionaries, each containing a matching row's data
            (excluding the auto-increment id field)

        Raises:
            Exception: If query fails
        """
        conditions = []
        values = []

        # Add field filters
        field_filters = {
            'model': model,
            'modelalias': modelalias,
            'hostname': hostname,
            'user': user,
            'project': project,
            'task': task,
            'note': note,
            'apikey_name': apikey_name
        }

        for field, value in field_filters.items():
            if value is not None:
                conditions.append(f'{field} = ?')
                values.append(value)

        # Add date range filters
        if date_from is not None:
            conditions.append('datetime >= ?')
            values.append(date_from)

        if date_to is not None:
            conditions.append('datetime <= ?')
            values.append(date_to)

        # Build WHERE clause
        where = ''
        if conditions:
            where = 'WHERE ' + ' AND '.join(conditions)

        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            try:
                conn.row_factory = sqlite3.Row  # Access columns by name
                cursor = conn.execute(
                    f'SELECT * FROM logs {where} ORDER BY datetime DESC, id DESC',
                    values
                )

                # Convert rows to list of dicts, excluding id field
                result = []
                for row in cursor:
                    row_dict = {key: row[key] for key in row.keys() if key != 'id'}
                    result.append(row_dict)

                return result
            finally:
                conn.close()
        except Exception as e:
            raise Exception(f"Failed to get rows: {e}") from e

    def get(self, model=None, modelalias=None, hostname=None, user=None,
            project=None, task=None, note=None, apikey_name=None,
            date_from=None, date_to=None) -> tuple[float, int, int, int]:
        """
        Get aggregated cost and token sums for rows matching specified criteria.

        Args:
            model: Filter by model name
            modelalias: Filter by model alias
            hostname: Filter by hostname
            user: Filter by user
            project: Filter by project
            task: Filter by task
            note: Filter by note
            apikey_name: Filter by API key name
            date_from: Filter by datetime >= this value (inclusive)
            date_to: Filter by datetime <= this value (inclusive)

        Returns:
            Tuple of (total_cost, total_input_tokens, total_output_tokens, row_count)

        Raises:
            Exception: If query fails
        """
        conditions = []
        values = []

        # Add field filters
        field_filters = {
            'model': model,
            'modelalias': modelalias,
            'hostname': hostname,
            'user': user,
            'project': project,
            'task': task,
            'note': note,
            'apikey_name': apikey_name
        }

        for field, value in field_filters.items():
            if value is not None:
                conditions.append(f'{field} = ?')
                values.append(value)

        # Add date range filters
        if date_from is not None:
            conditions.append('datetime >= ?')
            values.append(date_from)

        if date_to is not None:
            conditions.append('datetime <= ?')
            values.append(date_to)

        # Build WHERE clause
        where = ''
        if conditions:
            where = 'WHERE ' + ' AND '.join(conditions)

        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            try:
                cursor = conn.execute(
                    f'''SELECT 
                        COALESCE(SUM(cost), 0.0),
                        COALESCE(SUM(input_tokens), 0),
                        COALESCE(SUM(output_tokens), 0),
                        COUNT(*)
                    FROM logs {where}''',
                    values
                )
                result = cursor.fetchone()
                return (float(result[0]), int(result[1]), int(result[2]), int(result[3]))
            finally:
                conn.close()
        except Exception as e:
            raise Exception(f"Failed to get aggregated data: {e}") from e

    def get_cost(self) -> float:
        """
        Get sum of cost for rows matching default field values.

        Returns:
            Sum of cost field, or 0.0 if no matching rows

        Raises:
            Exception: If query fails
        """
        where, values = self._build_where_clause()

        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            try:
                cursor = conn.execute(
                    f'SELECT COALESCE(SUM(cost), 0.0) FROM logs {where}',
                    values
                )
                result = cursor.fetchone()[0]
                return float(result)
            finally:
                conn.close()
        except Exception as e:
            raise Exception(f"Failed to get cost sum: {e}") from e

    def get_input_tokens(self) -> int:
        """
        Get sum of input_tokens for rows matching default field values.

        Returns:
            Sum of input_tokens field, or 0 if no matching rows

        Raises:
            Exception: If query fails
        """
        where, values = self._build_where_clause()

        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            try:
                cursor = conn.execute(
                    f'SELECT COALESCE(SUM(input_tokens), 0) FROM logs {where}',
                    values
                )
                result = cursor.fetchone()[0]
                return int(result)
            finally:
                conn.close()
        except Exception as e:
            raise Exception(f"Failed to get input_tokens sum: {e}") from e

    def get_output_tokens(self) -> int:
        """
        Get sum of output_tokens for rows matching default field values.

        Returns:
            Sum of output_tokens field, or 0 if no matching rows

        Raises:
            Exception: If query fails
        """
        where, values = self._build_where_clause()

        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            try:
                cursor = conn.execute(
                    f'SELECT COALESCE(SUM(output_tokens), 0) FROM logs {where}',
                    values
                )
                result = cursor.fetchone()[0]
                return int(result)
            finally:
                conn.close()
        except Exception as e:
            raise Exception(f"Failed to get output_tokens sum: {e}") from e

    def stats(self, date_from=None, date_to=None) -> dict:
        """
        Generate summary statistics, optionally filtered to a date range.
        Includes rows with NULL values using 'NULL' as the key.

        Args:
            date_from: Optional lower bound for datetime (inclusive, 'YYYY-MM-DD HH:MM:SS')
            date_to:   Optional upper bound for datetime (inclusive, 'YYYY-MM-DD HH:MM:SS')

        Returns:
            Dictionary with four top-level keys:
            - 'by_all': {modelalias: cost, ...}
            - 'by_user': {user: {modelalias: cost, ...}, ...}
            - 'by_project': {project: {modelalias: cost, ...}, ...}
            - 'by_user_project': {"user / project": {modelalias: cost, ...}, ...}

            NULL values are represented as 'NULL' in the dictionary keys.

        Raises:
            Exception: If query fails
        """
        conditions = []
        values = []
        if date_from is not None:
            conditions.append('datetime >= ?')
            values.append(date_from)
        if date_to is not None:
            conditions.append('datetime <= ?')
            values.append(date_to)
        where = ('WHERE ' + ' AND '.join(conditions)) if conditions else ''

        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            try:
                result = {
                    'by_all': {},
                    'by_user': {},
                    'by_project': {},
                    'by_user_project': {}
                }

                # Grand total by modelalias
                cursor = conn.execute(
                    f'SELECT modelalias, SUM(cost) FROM logs {where} GROUP BY modelalias ORDER BY modelalias',
                    values
                )
                for modelalias, cost in cursor:
                    alias_key = modelalias if modelalias is not None else 'NULL'
                    result['by_all'][alias_key] = float(cost) if cost else 0.0

                # Cost by user and modelalias
                cursor = conn.execute(
                    f'SELECT user, modelalias, SUM(cost) FROM logs {where} GROUP BY user, modelalias ORDER BY user, modelalias',
                    values
                )
                for user, modelalias, cost in cursor:
                    user_key = user if user is not None else 'NULL'
                    if user_key not in result['by_user']:
                        result['by_user'][user_key] = {}
                    alias_key = modelalias if modelalias is not None else 'NULL'
                    result['by_user'][user_key][alias_key] = float(cost) if cost else 0.0

                # Cost by project and modelalias
                cursor = conn.execute(
                    f'SELECT project, modelalias, SUM(cost) FROM logs {where} GROUP BY project, modelalias ORDER BY project, modelalias',
                    values
                )
                for project, modelalias, cost in cursor:
                    project_key = project if project is not None else 'NULL'
                    if project_key not in result['by_project']:
                        result['by_project'][project_key] = {}
                    alias_key = modelalias if modelalias is not None else 'NULL'
                    result['by_project'][project_key][alias_key] = float(cost) if cost else 0.0

                # Cost by user/project combination and modelalias
                cursor = conn.execute(
                    f'SELECT user, project, modelalias, SUM(cost) FROM logs {where} GROUP BY user, project, modelalias ORDER BY user, project, modelalias',
                    values
                )
                for user, project, modelalias, cost in cursor:
                    user_key = user if user is not None else 'NULL'
                    project_key = project if project is not None else 'NULL'
                    key = f"{user_key} / {project_key}"
                    if key not in result['by_user_project']:
                        result['by_user_project'][key] = {}
                    alias_key = modelalias if modelalias is not None else 'NULL'
                    result['by_user_project'][key][alias_key] = float(cost) if cost else 0.0

                return result

            finally:
                conn.close()
        except Exception as e:
            raise Exception(f"Failed to generate stats: {e}") from e

    def get_months(self) -> list[str]:
        """
        Return sorted list of distinct 'YYYY-MM' strings present in the logs table.

        Raises:
            Exception: If query fails
        """
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            try:
                cursor = conn.execute(
                    "SELECT DISTINCT strftime('%Y-%m', datetime) FROM logs ORDER BY 1"
                )
                return [row[0] for row in cursor if row[0] is not None]
            finally:
                conn.close()
        except Exception as e:
            raise Exception(f"Failed to get months: {e}") from e

def fflt(value: float) -> str:
    return f"{value:.8f}".rstrip('0').rstrip('.')


def print_stats(all_stats: dict, by: str, byllm: bool):
    """Print cost statistics according to the selected grouping and LLM flag."""

    def section_total(groups: dict) -> str:
        return fflt(sum(sum(aliases.values()) for aliases in groups.values()))

    if by == 'all':
        aliases = all_stats['by_all']
        grand = fflt(sum(aliases.values()))
        if byllm:
            print(f"Total Cost: ${grand}")
            for alias, cost in aliases.items():
                print(f"  {alias}: ${fflt(cost)}")
        else:
            print(f"Total Cost: ${grand}")

    elif by == 'user':
        print("Cost by User:")
        groups = all_stats['by_user']
        for user, aliases in groups.items():
            print(f"  {user}: ${fflt(sum(aliases.values()))}")
            if byllm:
                for alias, cost in aliases.items():
                    print(f"    {alias}: ${fflt(cost)}")
        if len(groups) > 1:
            print(f"  Total: ${section_total(groups)}")

    elif by == 'project':
        print("Cost by Project:")
        groups = all_stats['by_project']
        for project, aliases in groups.items():
            print(f"  {project}: ${fflt(sum(aliases.values()))}")
            if byllm:
                for alias, cost in aliases.items():
                    print(f"    {alias}: ${fflt(cost)}")
        if len(groups) > 1:
            print(f"  Total: ${section_total(groups)}")

    elif by == 'user+project':
        print("Cost by User/Project:")
        groups = all_stats['by_user_project']
        for combo, aliases in groups.items():
            print(f"  {combo}: ${fflt(sum(aliases.values()))}")
            if byllm:
                for alias, cost in aliases.items():
                    print(f"    {alias}: ${fflt(cost)}")
        if len(groups) > 1:
            print(f"  Total: ${section_total(groups)}")


def get_args():
    parser = argparse.ArgumentParser(description='Show costs')
    parser.add_argument('file', type=str, help='Cost database file')
    parser.add_argument('--sum', action='store_true',
                        help='Output only the total cost as a float (ignores all other options)')
    parser.add_argument('--by', choices=['user', 'project', 'user+project', 'all'], default='all',
                        help='Grouping: user, project, user+project, or all (default: all)')
    parser.add_argument('--byllm', action='store_true',
                        help='Add per-LLM breakdown within the selected grouping')
    parser.add_argument('--bymonth', action='store_true',
                        help='Group output by year-month')
    args = parser.parse_args()
    return vars(args)


def main():
    config = get_args()
    costs = Log2Sqlite(config['file'])

    if config['sum']:
        total_cost, _, _, _ = costs.get()
        print(total_cost)
        return

    by = config['by']
    byllm = config['byllm']

    if config['bymonth']:
        months = costs.get_months()
        for month in months:
            year, mon = map(int, month.split('-'))
            last_day = calendar.monthrange(year, mon)[1]
            date_from = f"{month}-01 00:00:00"
            date_to = f"{month}-{last_day:02d} 23:59:59"
            all_stats = costs.stats(date_from=date_from, date_to=date_to)
            print(f"=== {month} ===")
            print_stats(all_stats, by, byllm)
            print()
    else:
        all_stats = costs.stats()
        print_stats(all_stats, by, byllm)


# Example usage
if __name__ == '__main__':
    main()
