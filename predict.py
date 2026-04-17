"""
predict_video.py — Detekcja obiektów z filmu przy użyciu YOLO

Instalacja:
    pip install ultralytics opencv-python

Użycie:
    python predict_video.py --model best.pt --source video.mp4
    python predict_video.py --model best.pt --source 0          # kamera
    python predict_video.py --model best.pt --source video.mp4 --conf 0.4 --save
"""

import argparse
import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Detekcja obiektów z filmu (YOLO)")
    parser.add_argument("--model", required=True, help="Ścieżka do modelu .pt")
    parser.add_argument("--source", required=True, help="Plik wideo lub nr kamery (0)")
    parser.add_argument(
        "--conf", type=float, default=0.50, help="Próg pewności (domyślnie: 0.25)"
    )
    parser.add_argument("--save", action="store_true", help="Zapisz wynikowe wideo")
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            "output1.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

    print("▶  Detekcja — naciśnij Q aby zakończyć")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = next(iter(model(frame, conf=args.conf, verbose=False, stream= True)))
        annotated = results.plot()

        if writer:
            writer.write(annotated)

        cv2.imshow("YOLO — detekcja", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
        print("✅  Zapisano: output.mp4")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
