"""Interactive console app to detect text bounding boxes using CRAFT."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from craft_text_detector import Craft
from PIL import Image, ImageDraw


@dataclass
class BoundingBox:
    """Simple rectangle defined by top-left and bottom-right points."""

    left: float
    top: float
    right: float
    bottom: float

    @classmethod
    def from_quadrilateral(cls, quad: Sequence[Sequence[float]]) -> "BoundingBox":
        xs = [point[0] for point in quad]
        ys = [point[1] for point in quad]
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
        boxes = [BoundingBox.from_quadrilateral(box) for box in raw_boxes]
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
