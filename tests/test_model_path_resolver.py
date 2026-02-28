"""Файл: tests/test_model_path_resolver.py
Тип: файл автотестов.
Назначение: проверяет безопасное и единообразное разрешение путей к production-моделям.
Связи: взаимодействует с resolver-утилитой и детекторами через публичные API.
Импортируемые внутренние модули:
- src.inference.model_path_resolver: используется для передачи данных или вызова связанной логики.
- src.detectors.yolo_detector: используется для передачи данных или вызова связанной логики.
- src.detectors.forbidden_items_detector: используется для передачи данных или вызова связанной логики.
- src.detectors.dms_detector: используется для передачи данных или вызова связанной логики."""

from pathlib import Path

import pytest

from src.detectors.dms_detector import DMSDetector
from src.detectors.forbidden_items_detector import ForbiddenItemsDetector
from src.detectors.yolo_detector import YOLODetector
from src.inference.model_path_resolver import get_default_model_root, resolve_model_path


class _ModelStub:
    names = {}


class _HubStub:
    def get_model(self, model_key, weights_path):
        return _ModelStub()


def test_resolve_model_path_returns_canonical_best_auto_model():
    resolved_path = resolve_model_path()
    model_root = get_default_model_root().resolve()

    assert Path(resolved_path).is_file()
    assert Path(resolved_path).name == "best_auto.pt"
    assert Path(resolved_path).resolve().parent == model_root


def test_resolve_model_path_ignores_cwd_fake_weight(tmp_path, monkeypatch):
    fake_weight = tmp_path / "best_auto.pt"
    fake_weight.write_text("fake", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    resolved_path = resolve_model_path("best_auto.pt")

    assert Path(resolved_path).resolve() != fake_weight.resolve()
    assert Path(resolved_path).resolve().parent == get_default_model_root().resolve()


def test_resolve_model_path_rejects_outside_allowed_root(tmp_path):
    outside_weight = tmp_path / "outside.pt"
    outside_weight.write_text("fake", encoding="utf-8")

    with pytest.raises(ValueError):
        resolve_model_path(str(outside_weight))


def test_resolve_model_path_raises_when_model_missing_in_allowed_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_model_path("missing.pt", model_root=tmp_path)


def test_all_detectors_use_same_canonical_model_path_and_ignore_cwd(tmp_path, monkeypatch):
    fake_weight = tmp_path / "best_auto.pt"
    fake_weight.write_text("fake", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    hub = _HubStub()

    yolo_detector = YOLODetector(hub=hub)
    forbidden_detector = ForbiddenItemsDetector(hub=hub)
    dms_detector = DMSDetector(hub=hub)
    expected_path = resolve_model_path()

    assert yolo_detector.model_path == expected_path
    assert forbidden_detector.model_path == expected_path
    assert dms_detector.model_path == expected_path
    assert expected_path != str(fake_weight.resolve())


@pytest.mark.parametrize("detector_cls, model_kwarg", [
    (YOLODetector, "model_name"),
    (ForbiddenItemsDetector, "model_name"),
    (DMSDetector, "model_path"),
])
def test_detectors_reject_model_path_outside_allowed_root(tmp_path, detector_cls, model_kwarg):
    outside_weight = tmp_path / "outside.pt"
    outside_weight.write_text("fake", encoding="utf-8")
    hub = _HubStub()

    with pytest.raises(ValueError):
        detector_cls(hub=hub, **{model_kwarg: str(outside_weight)})
