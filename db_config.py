import os

DB_URL = os.getenv('DATABASE_URL') or \
    f"postgresql://{os.getenv('PGUSER','postgres')}:{os.getenv('PGPASSWORD','newpassword')}@{os.getenv('PGHOST','localhost')}:{os.getenv('PGPORT','5432')}/{os.getenv('PGDATABASE','xfl_metrics')}"
