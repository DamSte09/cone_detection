"""
split_dataset.py — Podział danych YOLO na zbiory treningowy i testowy (80/20)

Struktura wejściowa:
    dataset/
    ├── images/   (obrazy: .jpg, .jpeg, .png, .bmp, .webp)
    └── labels/   (etykiety: .txt)

Struktura wyjściowa:
    dataset/
    ├── images/
    │   ├── train/
    │   └── test/
    └── labels/
        ├── train/
        └── test/

Użycie:
    python split_dataset.py                         # domyślnie folder ./dataset
    python split_dataset.py --dataset /ścieżka     # własna ścieżka
    python split_dataset.py --ratio 0.8 --seed 42  # własny podział i seed
    python split_dataset.py --copy                  # kopiuj zamiast przenosić
"""

import os
import random
import shutil
import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Podział datasetu YOLO na zbiory train/test."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset",
        help="Ścieżka do folderu dataset (domyślnie: ./dataset)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.8,
        help="Udział zbioru treningowego, np. 0.8 = 80%% train (domyślnie: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Ziarno losowości dla powtarzalności wyników (domyślnie: 42)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Kopiuj pliki zamiast je przenosić (domyślnie: przenoszenie)",
    )
    return parser.parse_args()


def find_image_label_pairs(images_dir: Path, labels_dir: Path):
    """Zwraca listę par (plik_obrazu, plik_etykiety) gdzie oba pliki istnieją."""
    pairs = []
    missing_labels = []

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = labels_dir / (img_path.stem + ".txt")

        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            missing_labels.append(img_path.name)

    return pairs, missing_labels


def create_output_dirs(dataset_dir: Path):
    """Tworzy foldery train/test w images i labels."""
    dirs = {
        "images_train": dataset_dir / "images" / "train",
        "images_test": dataset_dir / "images" / "test",
        "labels_train": dataset_dir / "labels" / "train",
        "labels_test": dataset_dir / "labels" / "test",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def move_or_copy(src: Path, dst_dir: Path, use_copy: bool):
    dst = dst_dir / src.name
    if use_copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), dst)


def split_dataset(dataset_dir: Path, ratio: float, seed: int, use_copy: bool):
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    # Walidacja struktury
    for d in (images_dir, labels_dir):
        if not d.exists():
            raise FileNotFoundError(f"Brak folderu: {d}")

    print(f"\n{'=' * 55}")
    print(f"  Dataset:  {dataset_dir.resolve()}")
    print(f"  Podział:  {int(ratio * 100)}% train / {int((1 - ratio) * 100)}% test")
    print(f"  Tryb:     {'kopiowanie' if use_copy else 'przenoszenie'}")
    print(f"  Seed:     {seed}")
    print(f"{'=' * 55}\n")

    # Znajdź pary obraz–etykieta
    pairs, missing = find_image_label_pairs(images_dir, labels_dir)

    if not pairs:
        print("❌  Nie znaleziono żadnych pasujących par obraz–etykieta.")
        return

    if missing:
        print(f"⚠️   Brak etykiet dla {len(missing)} obrazów (pominięto):")
        for name in missing[:10]:
            print(f"      - {name}")
        if len(missing) > 10:
            print(f"      ... i {len(missing) - 10} więcej")
        print()

    # Losowy podział
    random.seed(seed)
    random.shuffle(pairs)

    split_idx = int(len(pairs) * ratio)
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]

    print(f"  Łącznie par:   {len(pairs)}")
    print(f"  Train:         {len(train_pairs)}")
    print(f"  Test:          {len(test_pairs)}\n")

    # Tworzenie folderów
    dirs = create_output_dirs(dataset_dir)

    # Przenoszenie / kopiowanie plików
    action = "Kopiowanie" if use_copy else "Przenoszenie"

    print(f"⏳  {action} plików treningowych...")
    for img_path, lbl_path in train_pairs:
        move_or_copy(img_path, dirs["images_train"], use_copy)
        move_or_copy(lbl_path, dirs["labels_train"], use_copy)

    print(f"⏳  {action} plików testowych...")
    for img_path, lbl_path in test_pairs:
        move_or_copy(img_path, dirs["images_test"], use_copy)
        move_or_copy(lbl_path, dirs["labels_test"], use_copy)

    print(f"\n Gotowe!\n")
    print(f"  images/train/ → {len(train_pairs)} obrazów")
    print(f"  images/test/  → {len(test_pairs)} obrazów")
    print(f"  labels/train/ → {len(train_pairs)} etykiet")
    print(f"  labels/test/  → {len(test_pairs)} etykiet")
    print(f"\n{'=' * 55}\n")


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset)

    if not dataset_dir.exists():
        print(f"❌  Folder dataset nie istnieje: {dataset_dir.resolve()}")
        return

    if not (0 < args.ratio < 1):
        print("❌  Parametr --ratio musi być z przedziału (0, 1), np. 0.8")
        return

    split_dataset(
        dataset_dir=dataset_dir,
        ratio=args.ratio,
        seed=args.seed,
        use_copy=args.copy,
    )


if __name__ == "__main__":
    main()
