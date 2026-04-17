#!/usr/bin/env python3
"""
Ekstraktor klatek z pliku MP4
Używa OpenCV do wyodrębnienia klatek z filmu i zapisania ich jako obrazy.
"""

import cv2
import os
import argparse
import sys


def extract_frames(
    video_path: str,
    output_dir: str = "frames",
    every_n_frames: int = 1,
    max_frames: int = None,
    image_format: str = "jpg",
    quality: int = 95,
    resize: tuple = None,
    start_time: float = 0.0,
    end_time: float = None,
):
    """
    Wyodrębnia klatki z pliku wideo.

    Args:
        video_path:     Ścieżka do pliku MP4
        output_dir:     Katalog wyjściowy dla obrazów
        every_n_frames: Co ile klatek zapisywać (1 = każda, 30 = co 30. klatka)
        max_frames:     Maksymalna liczba klatek do zapisania (None = wszystkie)
        image_format:   Format obrazu: 'jpg', 'png', 'bmp', 'webp'
        quality:        Jakość JPEG/WebP (1-100)
        resize:         Rozmiar wyjściowy jako (szerokość, wysokość) lub None
        start_time:     Czas startu w sekundach
        end_time:       Czas końca w sekundach (None = do końca)
    """

    # Sprawdzenie pliku wejściowego
    if not os.path.exists(video_path):
        print(f"[BŁĄD] Plik nie istnieje: {video_path}")
        sys.exit(1)

    # Otwarcie pliku wideo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[BŁĄD] Nie można otworzyć pliku: {video_path}")
        sys.exit(1)

    # Pobranie metadanych wideo
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"\n{'=' * 50}")
    print(f"  Plik:       {os.path.basename(video_path)}")
    print(f"  Rozdzielczość: {width}x{height}")
    print(f"  FPS:        {fps:.2f}")
    print(f"  Klatki:     {total_frames}")
    print(f"  Czas trwania: {duration:.2f}s")
    print(f"{'=' * 50}\n")

    # Przeliczenie czasu na numery klatek
    start_frame = int(start_time * fps) if start_time > 0 else 0
    end_frame = int(end_time * fps) if end_time else total_frames

    end_frame = min(end_frame, total_frames)

    if start_frame >= end_frame:
        print("[BŁĄD] Czas startu musi być mniejszy niż czas końca.")
        sys.exit(1)

    # Ustawienie pozycji startowej
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Tworzenie katalogu wyjściowego
    os.makedirs(output_dir, exist_ok=True)

    # Ustawienia kompresji
    fmt = image_format.lower()
    if fmt == "jpg":
        ext = ".jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif fmt == "png":
        ext = ".png"
        # PNG: kompresja 0-9 (odwrotnie do jakości)
        compress = max(0, min(9, (100 - quality) // 10))
        params = [cv2.IMWRITE_PNG_COMPRESSION, compress]
    elif fmt == "webp":
        ext = ".webp"
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    elif fmt == "bmp":
        ext = ".bmp"
        params = []
    else:
        print(f"[BŁĄD] Nieznany format: {fmt}. Użyj: jpg, png, webp, bmp")
        sys.exit(1)

    # Szerokość numeru klatki w nazwie pliku (np. 00042)
    digits = len(str(total_frames))

    saved = 0
    current = start_frame

    print(f"Ekstrakcja klatek do: {os.path.abspath(output_dir)}/")
    print(
        f"Format: {ext.upper()[1:]} | Co {every_n_frames}. klatka | "
        f"Klatki {start_frame}–{end_frame}\n"
    )

    while current < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Pomiń klatki wg odstępu
        if (current - start_frame) % every_n_frames != 0:
            current += 1
            continue

        # Skalowanie
        if resize:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_LANCZOS4)

        # Zapis
        filename = os.path.join(output_dir, f"frame1_{current:0{digits}d}{ext}")
        cv2.imwrite(filename, frame, params)

        saved += 1
        # Pasek postępu
        progress = int(40 * (current - start_frame) / (end_frame - start_frame))
        bar = f"[{'█' * progress}{'░' * (40 - progress)}]"
        print(
            f"\r  {bar} {current}/{end_frame} | Zapisano: {saved}", end="", flush=True
        )

        if max_frames and saved >= max_frames:
            print(f"\n\n[INFO] Osiągnięto limit {max_frames} klatek.")
            break

        current += 1

    cap.release()
    print(f"\n\n{'=' * 50}")
    print(f"  Gotowe! Zapisano {saved} klatek do: {output_dir}/")
    print(f"{'=' * 50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Ekstraktor klatek z pliku MP4 (OpenCV)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
  # Każda 30. klatka (ok. 1 klatka/s przy 30fps)
  python extract_frames.py film.mp4 -n 30

  # Pierwsze 100 klatek w formacie PNG
  python extract_frames.py film.mp4 --format png --max 100

  # Klatki z przedziału 10s–60s, skalowane do 1280x720
  python extract_frames.py film.mp4 --start 10 --end 60 --resize 1280 720

  # Co 5. klatka, JPEG 80%, do katalogu "output"
  python extract_frames.py film.mp4 -n 5 -q 80 -o output
        """,
    )

    parser.add_argument("video", help="Ścieżka do pliku MP4")
    parser.add_argument(
        "-o",
        "--output",
        default="frames",
        help="Katalog wyjściowy (domyślnie: frames/)",
    )
    parser.add_argument(
        "-n",
        "--every",
        type=int,
        default=1,
        metavar="N",
        help="Zapisuj co N-tą klatkę (domyślnie: 1)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        metavar="N",
        help="Maks. liczba klatek do zapisania",
    )
    parser.add_argument(
        "--format",
        choices=["jpg", "png", "webp", "bmp"],
        default="jpg",
        help="Format wyjściowy (domyślnie: jpg)",
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=95,
        help="Jakość obrazu 1–100 (domyślnie: 95, dla JPG/WebP)",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        metavar=("W", "H"),
        help="Docelowy rozmiar: szerokość wysokość",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Czas startu w sekundach (domyślnie: 0)",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="Czas końca w sekundach (domyślnie: koniec)",
    )

    args = parser.parse_args()

    extract_frames(
        video_path=args.video,
        output_dir=args.output,
        every_n_frames=args.every,
        max_frames=args.max,
        image_format=args.format,
        quality=args.quality,
        resize=tuple(args.resize) if args.resize else None,
        start_time=args.start,
        end_time=args.end,
    )


if __name__ == "__main__":
    main()
