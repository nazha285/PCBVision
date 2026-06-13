from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/train-6/weights/best.pt")

# Predict on test images
results = model.predict(
    source="dataset/pcb-defect-dataset/test/images",
    save=True,
    conf=0.25
)

print("Prediction completed!")