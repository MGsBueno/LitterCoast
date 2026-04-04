from littercoast.config import DEFAULT_CLASSES, DetectionConfig, TrainingConfig


def test_detection_config_builds_class_to_id_mapping() -> None:
    config = DetectionConfig()

    assert config.classes == DEFAULT_CLASSES
    assert config.class_to_id["pet_bottle"] == 0
    assert config.class_to_id["fragment"] == len(DEFAULT_CLASSES) - 1


def test_training_config_computes_train_and_validation_roots() -> None:
    config = TrainingConfig()

    assert config.train_root.parts[-2:] == ("garbage_classification", "train")
    assert config.validation_root.parts[-2:] == ("garbage_classification", "val")
