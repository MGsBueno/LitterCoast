import sys
import types
import shutil
from pathlib import Path
from uuid import uuid4

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
TEST_RUNS_DIR = PROJECT_ROOT / ".tmp_test_runs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


if "ultralytics" not in sys.modules:
    ultralytics_module = types.ModuleType("ultralytics")

    class YOLO:  # pragma: no cover - lightweight stub for local unit tests
        def __init__(self, model_path):
            self.model_path = model_path

        def __call__(self, image):
            return []

        def train(self, **kwargs):
            return None

    ultralytics_module.YOLO = YOLO
    sys.modules["ultralytics"] = ultralytics_module


if "PIL" not in sys.modules:
    pil_module = types.ModuleType("PIL")

    class _ImageModule:
        @staticmethod
        def open(path):
            return path

    pil_module.Image = _ImageModule
    sys.modules["PIL"] = pil_module


if "qrcode" not in sys.modules:
    qrcode_module = types.ModuleType("qrcode")

    class _Constants:
        ERROR_CORRECT_L = "L"

    class _FakeImage:
        def save(self, output_path):
            Path(output_path).write_bytes(b"fake-qr")

    class QRCode:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.payload = None

        def add_data(self, data):
            self.payload = data

        def make(self, fit=True):
            return None

        def make_image(self, fill_color="black", back_color="white"):
            return _FakeImage()

    qrcode_module.constants = _Constants()
    qrcode_module.QRCode = QRCode
    sys.modules["qrcode"] = qrcode_module


@pytest.fixture
def workspace_dir():
    TEST_RUNS_DIR.mkdir(exist_ok=True)
    path = TEST_RUNS_DIR / uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
