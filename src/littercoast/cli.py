import argparse
import os
from pathlib import Path

from .config import AppEnvironment, InferenceConfig, QRCodeConfig, TrainingConfig
from .inference import InferencePipeline
from .qr_code import QRCodeGenerator
from .training import ModelTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LitterCoast project utilities")
    parser.add_argument(
        "--environment",
        choices=[environment.value for environment in AppEnvironment],
        help="Runtime environment preset used to resolve default paths",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Extract dataset and train YOLO")
    train_parser.add_argument("--archive", type=Path, help="Path to the dataset archive")
    train_parser.add_argument("--extracted-dir", type=Path, help="Directory where the archive will be extracted")
    train_parser.add_argument("--dataset-root", type=Path, help="Directory for generated train/val dataset")
    train_parser.add_argument("--dataset-yaml", type=Path, help="Path for the generated YOLO YAML")

    inference_parser = subparsers.add_parser("infer", help="Run inference for a directory of images")
    inference_parser.add_argument("--model", type=Path, help="Path to the trained model")
    inference_parser.add_argument("--images", type=Path, help="Directory containing images for inference")
    inference_parser.add_argument("--output", type=Path, help="JSON file for persisted predictions")

    qr_parser = subparsers.add_parser("qr", help="Generate QR code image")
    qr_parser.add_argument("--link", type=str, help="Link that will be encoded")
    qr_parser.add_argument("--output", type=Path, help="Output image path")

    api_parser = subparsers.add_parser("api", help="Launch the FastAPI server")
    api_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host used by the API server")
    api_parser.add_argument("--port", type=int, default=8000, help="Port used by the API server")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for local development")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.environment:
        os.environ["LITTERCOAST_ENVIRONMENT"] = args.environment

    if args.command == "train":
        config = TrainingConfig()
        if args.archive:
            config.dataset_archive_path = args.archive
        if args.extracted_dir:
            config.extracted_dir = args.extracted_dir
        if args.dataset_root:
            config.dataset_root = args.dataset_root
        if args.dataset_yaml:
            config.dataset_yaml_path = args.dataset_yaml
        ModelTrainer(training_config=config).run()
        return

    if args.command == "infer":
        config = InferenceConfig()
        if args.model:
            config.model_path = args.model
        if args.images:
            config.image_directory = args.images
        if args.output:
            config.predictions_path = args.output
        InferencePipeline(config).run()
        return

    if args.command == "qr":
        config = QRCodeConfig()
        if args.link:
            config.link = args.link
        if args.output:
            config.output_path = args.output
        QRCodeGenerator(config).generate()
        return

    if args.command == "api":
        import uvicorn

        uvicorn.run(
            "littercoast.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )


if __name__ == "__main__":
    main()
