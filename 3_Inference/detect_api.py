from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse , FileResponse
import os
import sys
def get_parent_dir(n=1):
    """ returns the n-th parent dicrectory of the current
    working directory """
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path



src_path = os.path.join(get_parent_dir(1), "2_Training", "src")
utils_path = os.path.join(get_parent_dir(1), "Utils")

sys.path.append(src_path)
sys.path.append(utils_path)


import argparse
from keras_yolo3.yolo import YOLO, detect_video
from PIL import Image
from timeit import default_timer as timer
from utils import load_extractor_model, load_features, parse_input, detect_object
import test
from io import BytesIO  # Add this line
import utils
import pandas as pd
import numpy as np
from Get_File_Paths import GetFileList
import random

app = FastAPI()

# Set up folder names for default values
data_folder = os.path.join(get_parent_dir(n=1), "Data")

image_folder = os.path.join(data_folder, "Source_Images")

image_test_folder = os.path.join(image_folder, "Test_Images")

detection_results_folder = os.path.join(image_folder, "Test_Image_Detection_Results")
detection_results_file = os.path.join(detection_results_folder, "Detection_Results.csv")

model_folder = os.path.join(data_folder, "Model_Weights")

model_weights = os.path.join(model_folder, "trained_weights_final.h5")
model_classes = os.path.join(model_folder, "data_classes.txt")

anchors_path = os.path.join(src_path, "keras_yolo3", "model_data", "yolo_anchors.txt")
# Initialize YOLO model
yolo = YOLO(
    **{
        "model_path": model_weights,
        "anchors_path": anchors_path,
        "classes_path": model_classes,
        "score": 0.25,
        "gpu_num": 1,
        "model_image_size": (416, 416),
    }
)

@app.post("/detect/")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    prediction, image = detect_object(yolo, image)
    # Save the processed image temporarily
    temp_file_path = "temp_annotated_image.jpg"
    image.save(temp_file_path, "JPEG")  # Save as JPEG

    # Return the file
    return FileResponse(temp_file_path, media_type='image/jpeg', filename="annotated_image.jpg")



def detect_object(yolo, image):
    prediction = yolo.detect_image(image)
    return prediction

def format_results(prediction, image):
    y_size, x_size = image.size
    formatted_predictions = []
    for single_prediction in prediction:
        formatted_predictions.append({
            "xmin": single_prediction[0],
            "ymin": single_prediction[1],
            "xmax": single_prediction[2],
            "ymax": single_prediction[3],
            "label": single_prediction[4],
            "confidence": single_prediction[5],
            "x_size": x_size,
            "y_size": y_size
        })
    return formatted_predictions

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

