from pathlib import Path
import cv2
from config.config import DATA_DIR

video_names = [
    'IMG_2251.MOV',
    'IMG_2252.MOV'
]


video_path = Path(DATA_DIR, "new_videos", video_names[1])
path_out = Path(DATA_DIR, "new_videos", "fix-" + video_names[1])

cap = cv2.VideoCapture(str(video_path))

# Definimos ancho y alto
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

totalFrame = 0
fourcc = cv2.VideoWriter_fourcc(*'MP4V')

writer = cv2.VideoWriter(str(path_out), fourcc, 30.0, (W, H), True)

rotate = True
center = (W / 2, H / 2)
angle = 180 + 90
scale = 1

M = cv2.getRotationMatrix2D(center, angle, scale)

print(f"Starting to read frames {video_names[0]}")
print(f"Shape: {W, H}")
while True:

    ret, frame = cap.read()

    
    if not ret:
        print("Cannot receive frames. Exiting...")
        break
    
    if rotate:
        frame = cv2.warpAffine(frame, M, (W, H))

    cv2.imshow("detections", frame)
    writer.write(frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()