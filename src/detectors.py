from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import tensorflow as tf

from config.config import MODELS_DIR
from src.utils import load_tf_model, load_tflite_model, download_models, pascal_voc_to_yolo


class TfDetection:

    def __init__(self, model_name: str):
        self.model_name = model_name

        self.model_dir = Path(MODELS_DIR / model_name)

        if not self.model_dir.exists():
            raise ValueError("model name not implemented")

        self.load_model()

    def load_model(self):
        if not self.model_dir.exists():
            print("model not exist in project directory, trying download")
            download_models(self.model_name)
            
        self.detector_model = load_tf_model(self.model_dir)


    def preprocess(self, frame: np.ndarray) -> tf.Tensor:

        image_np = np.array(frame)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]

        return input_tensor


    def filter_detection(self, frame: np.ndarray, detections: Dict) -> List:

        image = np.array(frame)
        scores = list(
            filter(lambda x: x>self.object_detection_trh , detections["detection_scores"])
            )

        # classes = detections["detection_classes"][:len(scores)]

        self.width = image.shape[1]
        self.height = image.shape[0]

        detections["detection_boxes"] = (
            detections["detection_boxes"] * np.array([self.height, self.width, self.height, self.width])
            ).astype(np.int64)

        filter_detections = {}
        filter_detections["bounding_box"]  = detections["detection_boxes"][:len(scores)]
        filter_detections["class_id"] = detections["detection_classes"][:len(scores)]
        filter_detections["score"] = detections["detection_scores"][:len(scores)]

        return filter_detections


    def _predict(self, frame: np.ndarray) -> np.ndarray:

        input_tensor = self.preprocess(frame)
        detections = self.detector_model(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        filter_detections = self.filter_detection(frame, detections)

        return filter_detections


    def predict(
        self, frame: np.ndarray, object_detection_trh: float=0.8
        ) -> np.ndarray:
        self.object_detection_trh = object_detection_trh
        detections = self._predict(frame)

        return detections

    def predict_yolo(
         self, frame: np.ndarray, object_detection_trh: float=0.8
        ) -> np.ndarray:
        self.object_detection_trh = object_detection_trh
        detections = self._predict(frame)

        detections["bounding_box"] = [
            pascal_voc_to_yolo(xyxy, self.width, self.height) for xyxy in  detections["bounding_box"]
            ]

        return detections

    

class TfliteDetection:

    def __init__(self, model_name: str):
        self.model_name = model_name

        self.model_dir = Path(MODELS_DIR / model_name)
        self.model_dir_name = Path(MODELS_DIR, model_name, "model.tflite" )

        if not self.model_dir.exists():
            raise ValueError("model name not implemented")

        self.load_model()
        self.input_details = self.detector_model.get_input_details()
        _, self.input_height, self.input_width, _ = self.input_details[0]['shape']

        self.output_details = self.detector_model.get_output_details()

        self.detector_model.allocate_tensors()

    def load_model(self):
        if not self.model_dir.exists():
            print("model not exist in project directory, trying download")
            download_models(self.model_name)
            
        self.detector_model = load_tflite_model(self.model_dir_name)


    def preprocess(self, frame: np.ndarray) -> tf.Tensor:

        im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(im_rgb,  (self.input_height, self.input_width))
        resized_img = resized_img / 255
        resized_img = resized_img[tf.newaxis, :]
        return resized_img

    def absolute_boundingbox(self, results, height, width):

        for obj in results:
            # Convert the bounding box figures from relative coordinates
            # to absolute coordinates based on the original resolution
            ymin, xmin, ymax, xmax = obj['bounding_box']
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)
            obj['bounding_box'] = [ymin, xmin, ymax, xmax]

        return results

    def _predict(self, frame: np.ndarray) -> np.ndarray:

        self.ori_input_height, self.ori_input_width = frame.shape[:2]
        resize_image = self.preprocess(frame)
        # self.detector_model.set_tensor(
        #     self.input_details[0]['index'], np.array(resize_image, dtype=np.uint8)
        #     )
        # tensor_index = self.detector_model.get_input_details()[0]['index']
        # input_tensor = self.detector_model.tensor(tensor_index)()[0]
        # input_tensor[:, :] = resize_image


        self.detector_model.set_tensor(
            self.input_details[0]['index'], np.array(resize_image, dtype=np.float32)
            )

         # run the inference
        self.detector_model.invoke()

        self.output_details = self.detector_model.get_output_details()

        boxes = np.squeeze(self.detector_model.get_tensor(self.output_details[0]['index']))
        classes = np.squeeze(self.detector_model.get_tensor(self.output_details[1]['index']))
        classes = classes.astype(np.int64)
        scores = np.squeeze(self.detector_model.get_tensor(self.output_details[2]['index']))     
        n_classes = int(np.squeeze(self.detector_model.get_tensor(self.output_details[3]['index'])))

        results = []
        for i in range(n_classes):
            if scores[i] >= self.object_detection_trh:
                result = {
                    'bounding_box': boxes[i],
                    'class_id': classes[i],
                    'score': scores[i]
                }
                results.append(result)

        if results:
            results = self.absolute_boundingbox(results, self.ori_input_height, self.ori_input_width)
        return results


    def predict(
        self, frame: np.ndarray, object_detection_trh: float=0.8
        ) -> np.ndarray:
        self.object_detection_trh = object_detection_trh
        detections = self._predict(frame)

        return detections