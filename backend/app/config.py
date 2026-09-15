import os
from pathlib import Path
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"   # backend/.env
load_dotenv(ENV_PATH)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(f"没读到 DATABASE_URL，检查文件是否存在: {ENV_PATH}")