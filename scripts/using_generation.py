from pathlib import Path
from pyzbar.pyzbar import decode

from src.generator import ImgGenerator
from config.config import DATA_DIR

use_device_size = (350, 640)
use_model_input = (410, 410)
generator = ImgGenerator(
    raw_input_size=use_device_size, output_size=use_model_input)

videos_name = ["ZEBRA-2022-10-24-11-05-11.mp4", "IPHONE-1762.MOV", "crop1-IPHONE-1763.MOV"]
video_path = Path(DATA_DIR, "raw_videos", videos_name[1])

generator.from_video(video_path)
