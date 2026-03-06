from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

ANIMAL_LABEL_KEYWORDS = {
    "animal",
    "bird",
    "cat",
    "cow",
    "deer",
    "dog",
    "elephant",
    "fox",
    "giraffe",
    "goat",
    "horse",
    "lion",
    "monkey",
    "pig",
    "rabbit",
    "sheep",
    "squirrel",
    "tiger",
    "wildlife",
    "zebra",
}

COCO_ANIMAL_CLASSES = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float


@dataclass(slots=True)
class ImageInference:
    has_animal: bool
    top_animal_confidence: float
    total_detections: int
    animal_detections: int
    detections: list[Detection]


class Detector(Protocol):
    def predict(self, image_path: Path) -> list[Detection]:
        pass


def _is_animal_label(label: str) -> bool:
    normalized = label.strip().lower()
    if normalized in COCO_ANIMAL_CLASSES:
        return True
    return any(keyword in normalized for keyword in ANIMAL_LABEL_KEYWORDS)


def is_animal_label(label: str) -> bool:
    return _is_animal_label(label)


class UltraLyticsDetector:
    def __init__(
        self,
        model_path: Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str | None = None,
    ) -> None:
        try:
            import numpy as np
            from models.common import DetectMultiBackend  # type: ignore[import-not-found]
            from utils.augmentations import letterbox  # type: ignore[import-not-found]
            from utils.general import check_img_size, cv2, non_max_suppression, scale_boxes  # type: ignore[import-not-found]
            from utils.torch_utils import select_device  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "YOLOv5 runtime is required for inference. Ensure vendor/yolov5 is on PYTHONPATH and torch is installed."
            ) from exc

        self._np = np
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required for inference. Install a CUDA-enabled torch build for GPU use."
            ) from exc
        self._torch = torch
        self._cv2 = cv2
        self._letterbox = letterbox
        self._nms = non_max_suppression
        self._scale_boxes = scale_boxes

        requested_device = (device or "").strip()
        if not requested_device and torch.cuda.is_available():
            requested_device = "cuda:0"
        try:
            self._device = select_device(requested_device)
        except Exception as exc:
            if requested_device.startswith("cuda"):
                self._device = select_device("cpu")
            else:
                raise RuntimeError(f"Failed to select inference device '{requested_device or 'auto'}'.") from exc

        self._device_name = str(self._device)
        self._model = DetectMultiBackend(str(model_path), device=self._device, dnn=False, data=None, fp16=False)
        self._imgsz = check_img_size((640, 640), s=self._model.stride)
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._model.warmup(imgsz=(1, 3, *self._imgsz))

    @property
    def device_name(self) -> str:
        return self._device_name

    def _prepare_image(self, image_path: Path) -> tuple[object, object]:
        im0 = self._cv2.imread(str(image_path))
        if im0 is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        im = self._letterbox(im0, self._imgsz, stride=self._model.stride, auto=self._model.pt)[0]
        im = im.transpose((2, 0, 1))[::-1]  # HWC BGR -> CHW RGB
        im = self._np.ascontiguousarray(im)
        return im0, im

    def _postprocess_predictions(self, pred: object, im0_list: list[object], input_shape: tuple[int, int]) -> list[list[Detection]]:
        names = self._model.names
        results: list[list[Detection]] = []

        for i, det in enumerate(pred):
            if det is None or len(det) == 0:
                results.append([])
                continue

            det[:, :4] = self._scale_boxes(input_shape, det[:, :4], im0_list[i].shape).round()
            detections: list[Detection] = []
            for *_, conf, cls in det:
                class_id = int(cls.item())
                confidence = float(conf.item())
                if isinstance(names, dict):
                    label = str(names.get(class_id, class_id))
                else:
                    label = str(names[class_id]) if class_id < len(names) else str(class_id)
                detections.append(Detection(label=label, confidence=confidence))
            results.append(detections)

        return results

    def predict_batch(self, image_paths: list[Path]) -> list[list[Detection]]:
        if not image_paths:
            return []

        prepared = [self._prepare_image(path) for path in image_paths]
        im0_list = [im0 for im0, _ in prepared]
        im_batch = self._np.stack([im for _, im in prepared], axis=0)

        tensor = self._torch.from_numpy(im_batch).to(self._model.device)
        tensor = tensor.half() if self._model.fp16 else tensor.float()
        tensor /= 255.0

        pred = self._model(tensor, augment=False, visualize=False)
        pred = self._nms(
            pred,
            conf_thres=self._conf,
            iou_thres=self._iou,
            classes=None,            agnostic=False,
            max_det=1000,
        )
        if not pred:
            return [[] for _ in image_paths]

        return self._postprocess_predictions(pred, im0_list, tensor.shape[2:])

    def predict(self, image_path: Path) -> list[Detection]:
        return self.predict_batch([image_path])[0]


def infer_image(detector: Detector, image_path: Path) -> ImageInference:
    detections = detector.predict(image_path)
    return infer_detections(detections)


def infer_detections(detections: list[Detection]) -> ImageInference:
    animal_confidences = [
        d.confidence for d in detections if _is_animal_label(d.label)
    ]
    top = max(animal_confidences) if animal_confidences else 0.0
    return ImageInference(
        has_animal=bool(animal_confidences),
        top_animal_confidence=top,
        total_detections=len(detections),
        animal_detections=len(animal_confidences),
        detections=detections,
    )
