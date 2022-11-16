
from typing import List, Tuple
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

from config.config import URL_MODELS, MODELS_DIR

def load_tf_model(model_path:Path):
    save_model_dir = Path(model_path, "saved_model")

    detect_fn = tf.saved_model.load(str(save_model_dir))

    return detect_fn

def load_tflite_model(model_path:Path):
    interpreter = tf.lite.Interpreter(str(model_path))

    return interpreter

def replace_char(chars, transform_dic):
  for s_char, r_char in transform_dic.items():
    chars = chars.replace(s_char, r_char)
  return chars

def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    
    plate = [] 
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate


def download_models(model_name: str):
    print("download fold from Google Drive ...")
    file_id = URL_MODELS[model_name]
    destination = Path(MODELS_DIR, model_name)

    gdown.download_folder(
        id=file_id,
        output=str(destination),
        use_cookies=False
    )

def resize_img(img, width, height):
    return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

def draw_first_plate(img, cropped_plate, all_plate_text, size:List=None):

    if size:
      width = int(size[0] * 0.6)
      height = int(size[1] * 0.6)

      x = int(img.shape[0] * 0.8)
      y = int(img.shape[1] * 0.1)

    else:
      # size for show plate
      width = int(img.shape[0] * 0.35)
      height = int(img.shape[1] * 0.15)

      x = int(img.shape[0] * 0.6)
      y = int(img.shape[1] * 0.1)

    plate = cropped_plate[0]
    plate_text = all_plate_text[0]

    plate = resize_img(plate, width, height)
    cv2.putText(img, plate_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
        1.5, (255, 10, 58), 3)
    img[y+10:y+10+height, x:x+width] = plate

    return img


def crop_center(img, dim):

    width, height = img.shape[1], img.shape[0]  #process crop width and height for max available dimension
    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 

    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img


def draw_detections(frame, detections, object_detection_trh):
    scores = list(
        filter(lambda x: x>object_detection_trh , detections["detection_scores"])
        )
    boxes = detections["detection_boxes"][:len(scores)]

    for box, score in zip(boxes, scores):
      cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), thickness = 2)
      cv2.putText(
        frame,
        f"score: {round(score, 2)}", (box[1] + 5, box[0] + 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
         )

    return frame

def relbox2absbox(boxes, height, width):
    return boxes * np.array([height, width, height, width])

def draw_tflite_detections(frame, results):
    for obj in results:
        ymin, xmin, ymax, xmax = obj['bounding_box']

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), 
                    (0, 0, 255), 2)
        y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        label = "{}: {:.2f}%".format(obj['class_id'],
            obj['score'] * 100)
        cv2.putText(frame, label, (xmin, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    return frame

def resize(img, objective_size:tuple, input_size:tuple):

    if objective_size[0] < objective_size[1]:
        objective_orientation = "vertical"
    else:
        objective_orientation = "horizontal"

    if input_size[0] < input_size[1]:
        input_orientation = "vertical"
    else:
        input_orientation = "horizontal"

    if objective_orientation == input_orientation:
        img = cv2.resize(img, objective_size)
        return img

    if objective_orientation == "vertical" and input_orientation == "horizontal":
        img = cv2.copyMakeBorder(
            img, int(input_size[1]/2), int(input_size[1]/2), 0, 0,cv2.BORDER_CONSTANT, None, value=0
            )
    else:
        img = cv2.copyMakeBorder(
            img, 0, 0, int(input_size[0]/2), int(input_size[0]/2), cv2.BORDER_CONSTANT, None, value=0
            )

    return cv2.resize(img, objective_size)
    
def transform2yolo(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[2])/2.0
    y = (box[1] + box[3])/2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def pascal_voc_to_yolo(x1y1x2y2, image_w, image_h):
    return [
        ((x1y1x2y2[3] + x1y1x2y2[1])/(2*image_w)),
        ((x1y1x2y2[2] + x1y1x2y2[0])/(2*image_h)),
        (x1y1x2y2[3] - x1y1x2y2[1])/image_w,
        (x1y1x2y2[2] - x1y1x2y2[0])/image_h,
        ]
