"""Interactive console app to detect text bounding boxes using CRAFT."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, cast

import numpy as np
from PIL import Image, ImageDraw


def _ensure_vgg_model_urls() -> None:
    """Backfill ``torchvision``'s removed ``model_urls`` attribute.

    Newer releases of ``torchvision`` no longer expose the ``model_urls``
    dictionary that older versions of :mod:`craft_text_detector` expect when
    importing ``torchvision.models.vgg``.  The library only needs a handful of
    VGG weight URLs, so when the attribute is missing we recreate it to keep the
    third-party dependency working without requiring users to downgrade
    ``torchvision``.
    """

    try:
        import torchvision.models.vgg as vgg_module  # type: ignore
    except Exception:  # pragma: no cover - torchvision might be unavailable.
        return

    if hasattr(vgg_module, "model_urls"):
        return

    vgg_module.model_urls = cast(  # type: ignore[attr-defined]
        Dict[str, str],
        {
            "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
            "vgg13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
            "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
            "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
            "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
            "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
            "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
            "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
        },
    )


_ensure_vgg_model_urls()


def _patch_craft_adjust_result_coordinates() -> None:
    """Harden ``craft_text_detector`` against irregular polygons.

    Some real-world images trigger a bug inside
    :mod:`craft_text_detector` where a handful of detected polygons have a
    different number of points than the rest.  The library attempts to
    coerce all polygons into a single :class:`numpy.ndarray`, which raises
    ``ValueError`` for these ragged inputs.  By monkey patching the helper
    used to rescale polygon coordinates we can gracefully skip malformed
    entries and keep the rest of the detections.
    """

    try:
        from craft_text_detector import craft_utils  # type: ignore
    except Exception:  # pragma: no cover - dependency might be missing.
        return

    original_adjust = craft_utils.adjustResultCoordinates

    def safe_adjust(
        polys: Iterable[Iterable[Sequence[float]]],
        ratio_w: float,
        ratio_h: float,
    ) -> List[List[List[float]]]:
        try:
            return original_adjust(polys, ratio_w, ratio_h)
        except ValueError:
            sanitized: List[List[List[float]]] = []
            if polys is None:  # type: ignore[redundant-expr]
                return sanitized
            for poly in polys:
                try:
                    arr = np.asarray(poly, dtype=float)
                except Exception:  # pragma: no cover - defensive.
                    continue
                if arr.ndim != 2 or arr.shape[1] != 2:
                    continue
                arr[:, 0] *= ratio_w
                arr[:, 1] *= ratio_h
                sanitized.append(arr.tolist())
            return sanitized

    craft_utils.adjustResultCoordinates = safe_adjust  # type: ignore[attr-defined]


_patch_craft_adjust_result_coordinates()

from craft_text_detector import Craft


@dataclass
class BoundingBox:
    """Simple rectangle defined by top-left and bottom-right points."""

    left: float
    top: float
    right: float
    bottom: float

    @classmethod
    def from_points(cls, points: Sequence[Sequence[float]]) -> "BoundingBox":
        sanitized = [point for point in points if len(point) >= 2]
        if not sanitized:
            raise ValueError("Tidak ada titik untuk membentuk bounding box.")

        xs = [point[0] for point in sanitized]
        ys = [point[1] for point in sanitized]
        return cls(min(xs), min(ys), max(xs), max(ys))

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.left, self.top, self.right, self.bottom)


def prompt_image_path() -> Path:
    while True:
        try:
            user_input = input("Masukkan path gambar yang ingin diproses: ").strip().strip('"')
        except EOFError:
            print("\nInput dihentikan. Keluar dari aplikasi.")
            sys.exit(0)

        if not user_input:
            print("Path tidak boleh kosong. Silakan coba lagi.\n")
            continue

        image_path = Path(user_input).expanduser().resolve()
        if not image_path.is_file():
            print(f"File tidak ditemukan: {image_path}. Silakan coba lagi.\n")
            continue

        return image_path


def build_output_path(image_path: Path) -> Path:
    stem = image_path.stem
    suffix = image_path.suffix or ".png"
    return image_path.with_name(f"{stem}_with_boxes{suffix}")


def load_image(image_path: Path) -> Image.Image:
    try:
        image = Image.open(image_path)
        image.load()
        return image
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"Gagal membuka gambar: {image_path}") from exc


def run_craft(image_path: Path) -> List[BoundingBox]:
    craft = Craft(output_dir=None, crop_type="poly", cuda=False)
    try:
        print("    Menjalankan deteksi teks dengan CRAFT...")
        prediction_result = craft.detect_text(str(image_path))
        raw_boxes = prediction_result.get("boxes") or []
        boxes = []
        for box in raw_boxes:
            try:
                boxes.append(BoundingBox.from_points(box))
            except (TypeError, ValueError):
                continue
        return boxes
    finally:
        craft.unload_craftnet_model()
        craft.unload_refinenet_model()


def draw_bounding_boxes(image: Image.Image, boxes: Sequence[BoundingBox]) -> Image.Image:
    annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)

    for box in boxes:
        draw.rectangle(box.to_tuple(), outline="red", width=2)

    return annotated


def main() -> None:
    print("============================")
    print("  Aplikasi Deteksi Teks CRAFT")
    print("============================\n")

    print("[1/5] Meminta input gambar dari pengguna...")
    image_path = prompt_image_path()
    output_path = build_output_path(image_path)

    print(f"[2/5] Memuat gambar dari {image_path}...")
    image = load_image(image_path)
    width, height = image.size
    print(f"    Ukuran gambar: {width}x{height} piksel")

    print("[3/5] Menyiapkan model CRAFT...")
    boxes = run_craft(image_path)
    print(f"    Ditemukan {len(boxes)} kandidat kotak teks.")

    print("[4/5] Menggambar bounding box pada gambar...")
    annotated = draw_bounding_boxes(image, boxes)

    print(f"[5/5] Menyimpan gambar baru ke {output_path}...")
    annotated.save(output_path)

    print("\nSelesai! Gambar dengan bounding box tersimpan di:")
    print(f"  {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProses dibatalkan oleh pengguna.")
        sys.exit(1)
