import psycopg2
import csv
import sys
import io
import os
import errno
import subprocess


class Database:

    def __init__(self):
        self.url = os.environ["DATABASE_URL"]
        self.connection = psycopg2.connect(self.url)
        self.cursor = self.connection.cursor()

    def __del__(self):
        self.rollback()
        self.connection.close()

    def mogrify(self, query, values):
        return self.cursor.mogrify(query, values).decode("utf8")
    
    def execute(self, query, values=None, returning=False):
        self.cursor.execute(query, values)
        if returning is False:
            return None
        else:
            try:
                return self.cursor.fetchone()[0]
            except (psycopg2.ProgrammingError, TypeError):
                return None            
    
    def query(self, query, values=None):
        self.cursor.execute(query, values)
        return self.cursor.fetchall()

    def commit(self):
        self.connection.commit()

    def rollback(self):
        self.connection.rollback()


