from pathlib import Path
import cv2
#import matplotlib.pyplot as plt
from src.detectors import TfDetection, TfliteDetection
from config.config import DATA_DIR, RESULTS_DIR
from src.utils import draw_detections, draw_tflite_detections

# model = TfDetection("ssd_mobilenet")
# model = TfliteDetection("ssd_mobilenet_lite_320x320")
# model = TfliteDetection("ssd_mobilenet")
# model = TfliteDetection("ssd_mobilenet_int8")
# model = TfliteDetection("ssd_plates")
# model = TfliteDetection("ssd_pets")
# model = TfliteDetection("ssd_mobilenet_sample")
model = TfliteDetection("ssd_mobilenet_pricetag")

# img = cv2.imread(str(DATA_DIR / "test1.jpg"))

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# croped, detections = model.predict(img)

# frame = draw_detections(img, detections, object_detection_trh=0.9)

# cv2.imshow("label detected", frame)

# #waits for user to press any key 
# #(this is necessary to avoid Python kernel form crashing)
# cv2.waitKey(0) 
  
# #closing all open windows 
# cv2.destroyAllWindows() 

device = 0
video_names = ["IPHONE-1766.MOV", "ZEBRA-2022-10-24-11-05-11.mp4"]
video_path = Path(DATA_DIR, "raw_videos", video_names[0])
path_out = RESULTS_DIR / "mall_out_video.mp4"
# cap = cv2.VideoCapture(device)
cap = cv2.VideoCapture(str(video_path))

# Definimos ancho y alto
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

totalFrame = 0
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
writer = cv2.VideoWriter(str(path_out), fourcc, 30.0, (W, H), True)

rotate = True
W = 320
H = 320
center = (W / 2, H / 2)
angle = 180
scale = 1

M = cv2.getRotationMatrix2D(center, angle, scale)

if not cap.isOpened():
    print(f"Cannot open {device}")
    exit()

print("Starting to read frames")
while True:

    ret, frame = cap.read()
    print(frame.shape)
    frame = cv2.resize(frame, (W, H))
    
    if not ret:
        print("Cannot receive frames. Exiting...")
        break

    if rotate:
        frame = cv2.warpAffine(frame, M, (W, H))

    detections = model.predict(frame)

    if detections:
        frame = draw_tflite_detections(frame, detections)


    cv2.imshow("detections", frame)
    writer.write(frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()


