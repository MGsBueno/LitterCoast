from pathlib import Path

import qrcode

from .config import QRCodeConfig


class QRCodeGenerator:
    """Creates QR code images from the configured link."""

    def __init__(self, config: QRCodeConfig | None = None) -> None:
        self.config = config or QRCodeConfig()

    def generate(self) -> Path:
        qr = qrcode.QRCode(
            version=self.config.version,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=self.config.box_size,
            border=self.config.border,
        )
        qr.add_data(self.config.link)
        qr.make(fit=True)

        image = qr.make_image(fill_color="black", back_color="white")
        image.save(self.config.output_path)
        print(f"QR code generated and saved as '{self.config.output_path}'")
        return self.config.output_path
