from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import create_model, BaseModel
from typing import Optional, List, Any, Dict, Union
import os
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor, Json
from dotenv import load_dotenv
from datetime import datetime
import json
import re
import argparse
import sqlite3
from pathlib import Path

# Load environment variables
load_dotenv()

class DatabaseConfig(BaseModel):
    """Database configuration model"""
    db_type: str = "postgres"  # or "sqlite"
    db_path: Optional[str] = None  # for SQLite
    db_name: Optional[str] = None
    db_user: Optional[str] = None
    db_password: Optional[str] = None
    db_host: Optional[str] = None
    db_port: Optional[str] = None

class DatabaseManager:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._conn = None
        
    def get_connection(self):
        if self._conn is None:
            if self.config.db_type == "postgres":
                self._conn = psycopg2.connect(
                    dbname=self.config.db_name,
                    user=self.config.db_user,
                    password=self.config.db_password,
                    host=self.config.db_host,
                    port=self.config.db_port
                )
            else:  # sqlite
                self._conn = sqlite3.connect(self.config.db_path)
                self._conn.row_factory = sqlite3.Row
        return self._conn
        
    def get_cursor(self):
        conn = self.get_connection()
        if self.config.db_type == "postgres":
            return conn.cursor(cursor_factory=RealDictCursor)
        return conn.cursor()
        
    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
            
    def execute(self, query, params=None):
        cursor = self.get_cursor()
        try:
            if self.config.db_type == "postgres":
                cursor.execute(query, params)
            else:
                # Convert Postgres-style placeholders to SQLite style
                query = query.replace("%s", "?")
                cursor.execute(query, params)
            return cursor
        except Exception as e:
            cursor.close()
            raise e

# Initialize database configuration
DB_CONFIG = DatabaseConfig(
    db_type=os.getenv("DB_TYPE", "postgres"),
    db_path=os.getenv("DB_PATH"),
    db_name=os.getenv("DB_NAME"),
    db_user=os.getenv("DB_USER"),
    db_password=os.getenv("DB_PASSWORD"),
    db_host=os.getenv("DB_HOST"),
    db_port=os.getenv("DB_PORT")
)

db = DatabaseManager(DB_CONFIG)

app = FastAPI()

# Mount the static files directory
base_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(base_dir, 'static')
app.mount("/static", StaticFiles(directory=static_dir), name="static")

def get_column_types(cursor, table_name):
    if DB_CONFIG.db_type == "postgres":
        query = """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        """
        cursor.execute(query, (table_name,))
        columns = cursor.fetchall()
        
        result = {}
        for column in columns:
            column_name, data_type, is_nullable = column['column_name'], column['data_type'], column['is_nullable']
            result[column_name] = {'type': data_type, 'nullable': is_nullable}
    else:  # sqlite
        query = f"PRAGMA table_info({table_name})"
        cursor.execute(query)
        columns = cursor.fetchall()
        
        result = {}
        for column in columns:
            column_name = column['name']
            data_type = column['type'].lower()
            is_nullable = 'YES' if not column['notnull'] else 'NO'
            
            # Map SQLite types to PostgreSQL types for consistency
            if 'int' in data_type:
                data_type = 'integer'
            elif 'char' in data_type or 'clob' in data_type or 'text' in data_type:
                data_type = 'text'
            elif 'real' in data_type or 'floa' in data_type or 'doub' in data_type:
                data_type = 'double precision'
            elif 'blob' in data_type:
                data_type = 'bytea'
            
            result[column_name] = {'type': data_type, 'nullable': is_nullable}
    
    return result

def build_json_path_query(column):
    parts = column.split('.')
    if len(parts) == 1:
        return sql.Identifier(column)
    else:
        base = sql.Identifier(parts[0])
        path = [sql.Literal(part) for part in parts[1:]]
        return sql.SQL('->').join([base] + path[:-1]) + sql.SQL('->>') + path[-1]

def flatten_json(data):
    flattened = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                flattened.update(flatten_json(value))
            else:
                flattened[key] = value
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                flattened.update(flatten_json(item))
            else:
                flattened[str(i)] = item
    else:
        flattened = data
    return flattened

# Update argparse
parser = argparse.ArgumentParser(description="SQL Dashboard with database and JSON handling options")
parser.add_argument("--db-type", choices=["postgres", "sqlite"], default="postgres", help="Database type to use (postgres or sqlite)")
parser.add_argument("--db-path", help="Path to SQLite database file (required for sqlite)")
parser.add_argument("--flatten-json", action="store_true", help="Flatten JSON columns into separate columns. If false, format as line-separated JSON.")
args = parser.parse_args()

# Override DB_CONFIG with command line arguments if provided
if args.db_type:
    DB_CONFIG.db_type = args.db_type
if args.db_path:
    DB_CONFIG.db_path = args.db_path

def process_json_data(value, flatten=False):
    if value is None:
        return value
    
    try:
        json_data = json.loads(value) if isinstance(value, str) else value
        if flatten:
            return flatten_json(json_data)
        else:
            return json.dumps(json_data, indent=2)  # Use indent=2 for pretty formatting
    except json.JSONDecodeError:
        return value

@app.get("/api/get-tables")
async def get_tables():
    cursor = None
    try:
        cursor = db.get_cursor()
        
        if DB_CONFIG.db_type == "postgres":
            query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """
            cursor.execute(query)
            all_tables = [row[0] for row in cursor.fetchall()]
        else:  # sqlite
            query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            cursor.execute(query)
            all_tables = [row[0] for row in cursor.fetchall()]
        
        non_empty_tables = []
        for table in all_tables:
            if DB_CONFIG.db_type == "postgres":
                count_query = sql.SQL("SELECT EXISTS(SELECT 1 FROM {} LIMIT 1)").format(sql.Identifier(table))
            else:
                count_query = f"SELECT EXISTS(SELECT 1 FROM {table} LIMIT 1)"
            cursor.execute(count_query)
            has_rows = cursor.fetchone()[0]
            if has_rows:
                non_empty_tables.append(table)
        
        return non_empty_tables
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while fetching table names.")
    finally:
        if cursor:
            cursor.close()

@app.get("/api/column-names")
async def get_column_names(table_name: str = Query(..., min_length=1)):
    cursor = None
    try:
        cursor = db.get_cursor()
        column_types = get_column_types(cursor, table_name)
        if not column_types:
            raise HTTPException(status_code=404, detail="Table not found or has no columns.")

        columns = []
        for col, info in column_types.items():
            column_info = {
                "name": col,
                "type": info['type'],
                "is_json": info['type'] in ('json', 'jsonb')
            }
            columns.append(column_info)

        return columns
    except Exception as e:
        print(f"An error occurred: {e}")
        if cursor:
            db.get_connection().rollback()
        raise HTTPException(status_code=500, detail="An unexpected error occurred while fetching column names.")
    finally:
        if cursor:
            cursor.close()

@app.get("/api/metrics-data")
async def get_metrics_data(
    table_name: str = Query(..., min_length=1),
    x_column: str = Query(None),
    y_column: str = Query(None),
    full_table: bool = Query(False),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000)
):
    cursor = None
    try:
        cursor = db.get_cursor()
        column_types = get_column_types(cursor, table_name)
        if not column_types:
            raise HTTPException(status_code=404, detail="Table not found or has no columns.")

        json_columns = [col for col, info in column_types.items() if info['type'] in ('json', 'jsonb')]
        offset = (page - 1) * page_size

        if full_table:
            if DB_CONFIG.db_type == "postgres":
                select_parts = [sql.Identifier(col) for col in column_types.keys()]
                query = sql.SQL("SELECT {} FROM {} OFFSET {} LIMIT {}").format(
                    sql.SQL(', ').join(select_parts),
                    sql.Identifier(table_name),
                    sql.Literal(offset),
                    sql.Literal(page_size)
                )
            else:
                columns = ', '.join(column_types.keys())
                query = f"SELECT {columns} FROM {table_name} LIMIT {page_size} OFFSET {offset}"
        else:
            if not x_column or not y_column:
                raise HTTPException(status_code=400, detail="X and Y columns must be specified when not fetching full table.")
            
            if DB_CONFIG.db_type == "postgres":
                x_select = build_json_path_query(x_column)
                y_select = build_json_path_query(y_column)
                query = sql.SQL("SELECT {} as x_value, {} as y_value FROM {} OFFSET {} LIMIT {}").format(
                    x_select, y_select,
                    sql.Identifier(table_name),
                    sql.Literal(offset),
                    sql.Literal(page_size)
                )
            else:
                # Simplified JSON path handling for SQLite
                x_col = x_column.split('.')[0]
                y_col = y_column.split('.')[0]
                query = f"SELECT {x_col} as x_value, {y_col} as y_value FROM {table_name} LIMIT {page_size} OFFSET {offset}"

        # Get total count
        if DB_CONFIG.db_type == "postgres":
            count_query = sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table_name))
        else:
            count_query = f"SELECT COUNT(*) FROM {table_name}"
        cursor.execute(count_query)
        total_count = cursor.fetchone()[0]

        cursor.execute(query)
        rows = cursor.fetchall()

        processed_rows = []
        for row in rows:
            if DB_CONFIG.db_type == "sqlite":
                row = dict(row)
            processed_row = {}
            for key, value in row.items():
                if key in json_columns and value is not None:
                    processed_row[key] = process_json_data(value, flatten=args.flatten_json)
                elif isinstance(value, datetime):
                    processed_row[key] = value.isoformat()
                else:
                    processed_row[key] = value
            
            if args.flatten_json:
                processed_row = flatten_json(processed_row)
            
            processed_rows.append(processed_row)

        return {
            "data": processed_rows,
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_count + page_size - 1) // page_size
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        if cursor:
            db.get_connection().rollback()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if cursor:
            cursor.close()

@app.get("/api/search")
async def search_database(
    table_name: str = Query(..., min_length=1),
    search_term: str = Query(..., min_length=1),
    columns: Optional[List[str]] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000)
):
    cursor = None
    try:
        cursor = db.get_cursor()

        column_types = get_column_types(cursor, table_name)
        if not column_types:
            raise HTTPException(status_code=404, detail="Table not found or has no columns.")

        if not columns:
            columns = list(column_types.keys())

        where_clauses = []
        for col in columns:
            if DB_CONFIG.db_type == "postgres":
                if column_types[col]['type'] in ('json', 'jsonb'):
                    where_clauses.append(
                        sql.SQL("{} ->> {} ILIKE {}").format(
                            sql.Identifier(col),
                            sql.Literal(''),
                            sql.Literal(f'%{search_term}%')
                        )
                    )
                elif column_types[col]['type'] in ('text', 'varchar', 'char'):
                    where_clauses.append(
                        sql.SQL("{} ILIKE {}").format(
                            sql.Identifier(col),
                            sql.Literal(f'%{search_term}%')
                        )
                    )
                else:
                    where_clauses.append(
                        sql.SQL("CAST({} AS TEXT) ILIKE {}").format(
                            sql.Identifier(col),
                            sql.Literal(f'%{search_term}%')
                        )
                    )
            else:  # sqlite
                where_clauses.append(f"{col} LIKE '%{search_term}%'")

        offset = (page - 1) * page_size

        if DB_CONFIG.db_type == "postgres":
            query = sql.SQL("SELECT * FROM {} WHERE {} OFFSET {} LIMIT {}").format(
                sql.Identifier(table_name),
                sql.SQL(" OR ").join(where_clauses),
                sql.Literal(offset),
                sql.Literal(page_size)
            )
            count_query = sql.SQL("SELECT COUNT(*) FROM {} WHERE {}").format(
                sql.Identifier(table_name),
                sql.SQL(" OR ").join(where_clauses)
            )
        else:
            where_clause = " OR ".join(where_clauses)
            query = f"SELECT * FROM {table_name} WHERE {where_clause} LIMIT {page_size} OFFSET {offset}"
            count_query = f"SELECT COUNT(*) FROM {table_name} WHERE {where_clause}"

        cursor.execute(count_query)
        total_count = cursor.fetchone()[0]

        cursor.execute(query)
        results = cursor.fetchall()

        processed_results = []
        for row in results:
            if DB_CONFIG.db_type == "sqlite":
                row = dict(row)
            processed_row = {}
            for key, value in row.items():
                if column_types[key]['type'] in ('json', 'jsonb') and value is not None:
                    processed_row[key] = process_json_data(value, flatten=args.flatten_json)
                elif isinstance(value, datetime):
                    processed_row[key] = value.isoformat()
                else:
                    processed_row[key] = value
            
            if args.flatten_json:
                processed_row = flatten_json(processed_row)
            
            processed_results.append(processed_row)

        return {
            "data": processed_results,
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_count + page_size - 1) // page_size
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        if cursor:
            db.get_connection().rollback()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if cursor:
            cursor.close()

@app.get("/")
async def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
