from pathlib import Path

from  src.autolabel import AutoLabel
from src.generator import ImgGenerator
from src.detectors import TfDetection

from config.config import DATA_DIR

def main():

    # expected device size input
    input_size = (640, 320)

    # output size for train
    output_size = (640, 640)

    generator = ImgGenerator(
        input_size, output_size, p_value=0.1, quality_selector=False, use_resize=True
        )

    detector_1 = TfDetection("ssd_mobilenet")

    autolabeling = AutoLabel(generator=generator, detectors=[detector_1])

    videos_dir = Path(DATA_DIR, "new_videos")
    autolabeling.from_video(videos_dir)


if __name__ == "__main__":
    main()