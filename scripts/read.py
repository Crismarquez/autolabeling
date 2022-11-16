from pathlib import Path
import contextlib
import cv2
#import matplotlib.pyplot as plt
from pyzbar.pyzbar import decode
from src.detectors import TfDetection, TfliteDetection
from config.config import DATA_DIR, RESULTS_DIR

model = TfliteDetection("ssd_mobilenet_pricetag")

img_name = "20220809_194200-183_jpg.rf.65f29298017cf19d1639f0bc099c7034.jpg"
file_path = Path(DATA_DIR, "labeled", "images", img_name)

img = cv2.imread(str(file_path))

print(img.shape)

# img = cv2.resize(img, (320, 320))

detections = model.predict(img)

crops_img = []
for det in detections:
    bbox = det["bounding_box"]
    _img = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    crops_img.append(_img)


# Displaying the image 
cv2.imshow("img", _img)
  
#waits for user to press any key 
#(this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0) 
  
#closing all open windows 
cv2.destroyAllWindows() 

results = decode(_img)



print(results)

