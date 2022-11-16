from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
RESULTS_DIR = Path(BASE_DIR, "results")
STORE_DIR = Path(BASE_DIR, "store")
MODELS_DIR = Path(STORE_DIR, "models")
AUTOLABEL_DIR = Path(DATA_DIR, "autolabeled")

# Create dirs
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


URL_MODELS = {
    "mobilenet_v1": "1nw_uPnm14hzFVjp1E34oK1ObyzBdvk9x",
    "mobilenet_v2": "1a7U05ttb693hx7CR82Sn__P-t_NIUX2B",
    "onnx":"1lunDZ1nlc5On3AUst_LGFVaNpJhg24Rt",
    "pt":"1CS5pJ-lqJgY1-czcU0atflhzNizZ9qCL"
}
