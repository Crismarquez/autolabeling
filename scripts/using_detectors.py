from pathlib import Path
import cv2
#import matplotlib.pyplot as plt
from src.detectors import TfDetection, TfliteDetection
from config.config import DATA_DIR, RESULTS_DIR

model = TfDetection("ssd_mobilenet")

#model = TfliteDetection("ssd_mobilenet_pricetag")

img = cv2.imread(str(Path(DATA_DIR, "images_generated", "1.jpg")))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

yolo_detections = model.predict_yolo(img, object_detection_trh=0.7)

print(yolo_detections)

with open(str(DATA_DIR / '0_auto.txt'), 'w') as f:
    for box in yolo_detections["bounding_box"]:
        line = ["0"] + [str(x) for x in box]
        line = " ".join(line)
        f.write(line)
        f.write('\n')

f.close()

