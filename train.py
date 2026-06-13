from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset/pcb-defect-dataset/data.yaml",
        epochs=20,
        imgsz=640,
        batch=2,
        workers=0,
        plots=False
    )