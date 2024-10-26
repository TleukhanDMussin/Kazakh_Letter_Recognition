# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8n.yaml')  # build a new model from YAML
#model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    model = YOLO('yolov8s.pt')  # build from YAML and transfer weights

    # Train the model
    results = model.train(data='Jamilya.yaml', epochs=100, batch = 4)

