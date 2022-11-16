from pathlib import Path
import cv2
import numpy as np

from pyzbar.pyzbar import decode

from src.utils import resize
from config.config import DATA_DIR


class ImgGenerator:

    def __init__(
        self, raw_input_size=(350, 640),
        output_size=(320, 320), 
        p_value: float=0.1,
        quality_selector=False,
        use_resize=True):
        
        self.raw_input_size = raw_input_size
        self.output_size = output_size
        self.p_value = p_value
        self.quality_selector = quality_selector
        self.use_resize = use_resize

        self.result_dir = Path(DATA_DIR, "images_generated")
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def from_video(self, video_path: Path):

        cap = cv2.VideoCapture(str(video_path))

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        count_img_save = 0
        p = self.p_value
        
        while True:
            ret, frame = cap.read()
            take_img = np.random.choice([True, False], p=[p, 1-p])
            if not ret:
                print("Cannot receive frames. Exiting...")
                break
            
            if take_img:
                if self.quality_selector:
                    results = decode(frame)
                    if len(results) >0:
                        raw_data = results[0].data
                        if len(raw_data)!= 22:
                            continue

                    else:
                        continue

                if self.use_resize:
                    frame = resize(
                        frame, self.raw_input_size, (W, H)
                        )

                    frame = cv2.resize(frame, self.output_size)

                img_dir = Path(
                    self.result_dir, video_path.stem + "-" + str(count_img_save) + ".jpg"
                    )

                cv2.imwrite(str(img_dir), frame)
                count_img_save += 1
