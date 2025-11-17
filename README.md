# YOLOv8 Object Detection for Real-Time Chess Board State Recognition
A robust computer vision system developed to localize and classify 13 distinct chess pieces on a board using Transfer Learning and the YOLOv8 Nano (YOLOv8n) model.The primary engineering goal was to achieve high accuracy despite training on a small, low-diversity initial dataset.

Final Performance (Unbiased Test Set)
The final evaluation was conducted on a dedicated, unseen test set (10% of the final data), confirming the model's ability to generalize well and avoid overfitting.

Final mAP50-95 Score: 79% This score proves high-precision accuracy, showing the model correctly identified the piece class and located its bounding box with tight precision.

Final mAP50 Score: 98.9%  The model exhibits near-perfect reliability in simply finding and classifying the pieces, minimizing False Positives and Negatives.

Conclusion: The test set performance was within $0.5\%$ of the validation score, demonstrating successful generalization due to the aggressive data strategy employed.

Data Strategy and Augmentation Pipeline: This project successfully overcame the limitations of the initial small dataset by implementing a robust preprocessing and augmentation strategy.

Data Source: Chess Pieces Dataset (Available on Kaggle: https://www.kaggle.com/datasets/imtkaggleteam/chess-pieces-detection-image-dataset/data).

Data Cleaning: The raw dataset was filtered to remove 3000+ duplicated frames, retaining 300 unique, original images.

Augmentation (Roboflow Concepts): Aggressive techniques (Rotation, Scaling, Brightness, Horizontal Flip) were applied to artificially inflate the training set to 1613 images. This was the primary method used to prevent overfitting and simulate diverse real-world conditions.

Model Selection: The compact YOLOv8n model was chosen to intentionally limit its capacity for memorization, forcing the model to learn only generalizable features.

**Setup and Reproduction**

The project was developed in Python using the Ultralytics framework.

1. Requirements
Clone this repository and install all necessary dependencies using the provided file:

git clone https://github.com/Shreevarthini/Object-detection.git
cd Object-detection
pip install -r requirements.txt

2. Training and Evaluation
All steps for data download, training, and final evaluation are documented in the chess_detection.ipynb Jupyter Notebook.

To load the final model and run inference:

from ultralytics import YOLO 

Load the best performing weights from the training run

model = YOLO('runs/detect/train/weights/best.pt') 

Run prediction on a new image

model.predict(source='path/to/new_image.jpg', save=True, conf=0.70)
