__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
#NOTE - this is necessary for compatibility with the mill


import optuna
from optuna_dashboard import run_server

run_server("sqlite:///playlist_model.db")