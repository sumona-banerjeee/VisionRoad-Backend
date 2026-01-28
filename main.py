import torch

# Monkeypatch torch.load BEFORE any imports that load models
# Fixes PyTorch 2.6+ weights_only=True default breaking YOLO model loading
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from app import create_app
import uvicorn

app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port = 8000)
