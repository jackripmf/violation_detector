"""Тесты нормализации имён детекторов."""

from src.utils.config.detector_aliases import normalize_detector_list


def test_normalize_detector_aliases_to_canonical_names():
    detectors = [
        "cvdetector",
        "yolodetector",
        "movementdetector",
        "forbiddendetector",
        "dmsdetector",
        "dark",
    ]

    assert normalize_detector_list(detectors) == [
        "cv",
        "yolo",
        "movement",
        "forbidden",
        "dms",
        "dark",
    ]


def test_normalize_detector_list_removes_duplicates_and_empty_values():
    detectors = ["cv", "cvdetector", "  ", "YOLO", "yolodetector", "darkdetector"]

    assert normalize_detector_list(detectors) == ["cv", "yolo", "dark"]
