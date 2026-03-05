import time
import builtins

_boot_start = time.time()
builtins._boot_start = _boot_start  # Share with app lifespan

from app import create_app
import uvicorn

app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
