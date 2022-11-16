from pathlib import Path
from typing import List
import sys
import logging
import shutil
import cv2
from config.config import DATA_DIR, AUTOLABEL_DIR

logging.basicConfig(
    format = '%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

class AutoLabel:
    def __init__(self, generator, detectors: List, object_detection_trh: float=0.7):

        self.generator = generator
        self.detectors = detectors
        self.object_detection_trh = object_detection_trh

        self.images_labeled = Path(AUTOLABEL_DIR, "labeled", "images")
        self.text_labeled = Path(AUTOLABEL_DIR, "labeled", "labels")
        self.not_labeled = Path(AUTOLABEL_DIR, "not_labeled")

    def from_video(self, video_dir: Path):
        
        logger.info('Image generation ...')
        for file in video_dir.iterdir():
            self.generator.from_video(file)

        logger.info('Auto-labeling ...')
        image_generated_dir = Path(DATA_DIR, "images_generated")

        for file in image_generated_dir.iterdir():

            name = file.stem
            name_ext = file.name
            img = cv2.imread(str(file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # yolo format detection
            yolo_detections = self.detectors[0].predict_yolo(img, self.object_detection_trh)

            if len(yolo_detections["bounding_box"])>1:
                txt_file = name + ".txt"

                with open(str(self.text_labeled/ txt_file), 'w') as f:
                    for box in yolo_detections["bounding_box"]:
                        line = ["0"] + [str(x) for x in box]
                        line = " ".join(line)
                        f.write(line)
                        f.write('\n')

                f.close()

                file_dir_dest = Path(self.images_labeled, name_ext)
                shutil.copyfile(file, file_dir_dest)
            
            else:

                file_dir_dest = Path(self.not_labeled, name_ext)
                shutil.copyfile(file, file_dir_dest)