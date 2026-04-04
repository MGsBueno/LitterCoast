import sys
import types
import shutil
import inspect
from enum import Enum
from pathlib import Path
from typing import get_type_hints
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


if "pydantic" not in sys.modules:
    pydantic_module = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **kwargs):
            annotations = get_type_hints(self.__class__)
            for name, annotation in annotations.items():
                default = getattr(self.__class__, name, None)
                value = kwargs.get(name, default)

                enum_type = None
                if inspect.isclass(annotation) and issubclass(annotation, Enum):
                    enum_type = annotation
                elif hasattr(annotation, "__args__"):
                    for option in annotation.__args__:
                        if inspect.isclass(option) and issubclass(option, Enum):
                            enum_type = option
                            break

                if enum_type and isinstance(value, str):
                    value = enum_type(value)

                setattr(self, name, value)

        def model_dump(self):
            return self.__dict__.copy()

    def Field(default=None, **kwargs):
        return default

    pydantic_module.BaseModel = BaseModel
    pydantic_module.Field = Field
    sys.modules["pydantic"] = pydantic_module


if "fastapi" not in sys.modules:
    fastapi_module = types.ModuleType("fastapi")

    class FastAPI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.routes = []

        def get(self, path, response_model=None):
            def decorator(func):
                self.routes.append(
                    {
                        "method": "GET",
                        "path": path,
                        "response_model": response_model,
                        "endpoint": func,
                    }
                )
                return func

            return decorator

        def post(self, path, response_model=None):
            def decorator(func):
                self.routes.append(
                    {
                        "method": "POST",
                        "path": path,
                        "response_model": response_model,
                        "endpoint": func,
                    }
                )
                return func

            return decorator

    fastapi_module.FastAPI = FastAPI
    sys.modules["fastapi"] = fastapi_module


@pytest.fixture
def workspace_dir():
    TEST_RUNS_DIR.mkdir(exist_ok=True)
    path = TEST_RUNS_DIR / uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _serialize_value(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return {key: _serialize_value(item) for key, item in value.__dict__.items()}
    return value


class ApiResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return _serialize_value(self._payload)


class ApiTestClient:
    def __init__(self, app):
        self.app = app

    def get(self, path):
        endpoint = self._resolve_endpoint(path, "GET")
        return ApiResponse(200, endpoint())

    def post(self, path, json=None):
        endpoint = self._resolve_endpoint(path, "POST")
        signature = inspect.signature(endpoint)
        parameters = list(signature.parameters.values())

        if not parameters:
            return ApiResponse(200, endpoint())

        request_parameter = parameters[0]
        request_model = get_type_hints(endpoint)[request_parameter.name]
        request_payload = request_model(**(json or {}))
        return ApiResponse(200, endpoint(request_payload))

    def _resolve_endpoint(self, path, method):
        for route in self.app.routes:
            route_path = route["path"] if isinstance(route, dict) else route.path
            route_methods = {route["method"]} if isinstance(route, dict) else set(route.methods)
            route_endpoint = route["endpoint"] if isinstance(route, dict) else route.endpoint

            if route_path == path and method in route_methods:
                return route_endpoint

        raise AssertionError(f"route not found: {method} {path}")


@pytest.fixture
def api_client():
    from littercoast.api import app

    return ApiTestClient(app)
