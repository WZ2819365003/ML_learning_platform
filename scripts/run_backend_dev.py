import os
import sys
from pathlib import Path

import uvicorn


BACKEND_ROOT = Path(__file__).resolve().parents[1] / "ml_platform"
os.chdir(BACKEND_ROOT)
sys.path.insert(0, str(BACKEND_ROOT))


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        loop="asyncio",
        log_level="warning",
    )
