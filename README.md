# Kazakh Character Detection Using YOLO

This repository contains code for training and evaluating a **YOLO-based object detection model** for detecting Kazakh characters from images. The project was implemented for ELCE 455 (Machine Learning) course.

## Project Overview

The purpose of this project is to detect and classify handwritten Kazakh characters using the YOLO algorithm. The model has been trained to recognize multiple characters, including those specific to the Kazakh language. It uses a modified YOLO model tailored for detecting grayscale images of handwritten characters.

### Key Features
- **Custom YOLO Implementation**: Uses YOLO for efficient and accurate object detection.
- **Custom Dataset**: Trained on a dataset of grayscale images of Kazakh characters, specifically designed for this project.
- **Real-time Detection**: Capable of detecting and classifying Kazakh characters in real-time.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Configuration](#configuration)
- [Results](#results)
- [License](#license)

## Installation

### Requirements
- Python 3.x
- PyTorch
- OpenCV
- Ultralytics YOLO package

### Install Dependencies
First, clone the repository:
```bash
git clone https://github.com/your-username/Kazakh_Letter_Recognition.git
cd Kazakh_Letter_Recognition.git
```

Install the dependencies using pip:
```bash
pip install -r requirements.txt
```

This will install the necessary libraries, including PyTorch, OpenCV, and the YOLO package by Ultralytics.

## Dataset

The dataset used for training and validation consists of grayscale images of handwritten Kazakh characters. The dataset structure is defined in the **`Jamilya.yaml`** file:
- **Train/Val/Test Paths**:
  - `train/images`: Path to training images.
  - `val/images`: Path to validation images.
  - `test/images` (optional): Path to test images.
- **Classes**: The dataset contains the following classes:
  1. ә
  2. ғ
  3. һ
  4. і
  5. қ
  6. ң
  7. ө
  8. ұ
  9. ү

### Example `Jamilya.yaml` File
```yaml
path: D:\JamilyaYOLO\Kazakh_Dataset_Gray\
train: train\images
val: val\images
test: test\images

names:
  0: ae
  1: gh
  2: h
  3: ii
  4: kh
  5: nh
  6: oe
  7: uh
  8: uu
```

## Usage

### Training
To train the YOLO model on the custom Kazakh character dataset, use the provided **`train.py`** script:
```bash
python train.py --data Jamilya.yaml --epochs 100 --batch-size 16 --img-size 640
```

This script will train the YOLO model on the specified dataset and save the model weights.

### Evaluation
Use the **`main.py`** script to evaluate the model's performance:
```bash
python main.py --data Jamilya.yaml --weights runs/train/exp/weights/best.pt --img-size 640
```

This script will load the trained model and evaluate its performance on the validation dataset.

### Inference
To run inference on a single image or a batch of images:
```bash
python main.py --source test/images --weights runs/train/exp/weights/best.pt --img-size 640 --conf 0.25
```

This will use the trained YOLO model to detect Kazakh characters in the specified image(s).

## Configuration

The model configuration is defined in the **`Jamilya.yaml`** file, which includes the paths to the dataset and the class names. Adjust the file paths to match your dataset location before running the training or evaluation scripts.

## Results

The model achieves accurate detection and classification of Kazakh characters with the YOLO architecture, demonstrating robust performance on handwritten characters. The results include metrics such as:
- **Precision**
- **Recall**
- **mAP (Mean Average Precision)**

These metrics can be visualized using the output logs generated during training.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
