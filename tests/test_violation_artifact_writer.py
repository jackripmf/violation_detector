"""Файл: tests/test_violation_artifact_writer.py
Тип: файл автотестов.
Назначение: проверяет запись артефактов нарушений вне orchestration-логики ViolationManager.
Связи: взаимодействует с violation_artifact_writer и file_manager через публичные методы."""

import numpy as np

from src.processing.violation_artifact_writer import ViolationArtifactWriter
from src.utils.io.file_manager import FileManager


def test_violation_artifact_writer_saves_dms_artifacts(monkeypatch, tmp_path):
    writer = ViolationArtifactWriter(FileManager(str(tmp_path)))
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    saved = {"image": 0, "report": 0}

    monkeypatch.setattr("cv2.imwrite", lambda path, image: saved.__setitem__("image", saved["image"] + 1) or True)
    monkeypatch.setattr(
        writer.file_manager,
        "save_violation_report",
        lambda *args, **kwargs: saved.__setitem__("report", saved["report"] + 1) or True,
    )

    ok = writer.write_dms_artifacts(
        annotated_frame=frame,
        img_path=str(tmp_path / "frame.jpg"),
        img_filename="frame.jpg",
        violation_info={"violation_type": "dms"},
        report_filename="frame.txt",
    )

    assert ok is True
    assert saved["image"] == 1
    assert saved["report"] == 1
