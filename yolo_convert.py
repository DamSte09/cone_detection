from ultralytics import YOLO
import blobconverter



model = YOLO("best.pt")
model.export(format="onnx", imgsz=416,opset=12)


blob_path = blobconverter.from_onnx(
    model="./best.onnx",
    data_type="FP16",
    shaves=6,  # dla OAK-D Lite: 6 shaves to optymalnie
    output_dir="./blob",
)
print(blob_path)