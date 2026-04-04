import os
from pathlib import Path


def load_env_file(env_path: Path | None = None) -> None:
    path = env_path or Path(".env")
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default))


def get_env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def get_env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def get_env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))
