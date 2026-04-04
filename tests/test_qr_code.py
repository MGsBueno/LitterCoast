from pathlib import Path

from littercoast.config import QRCodeConfig
from littercoast.qr_code import QRCodeGenerator


def test_qr_code_generator_creates_output_file(workspace_dir: Path) -> None:
    output_path = workspace_dir / "qr.png"
    config = QRCodeConfig(link="https://example.com", output_path=output_path)

    result = QRCodeGenerator(config).generate()

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
