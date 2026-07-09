from pathlib import Path

DATA_ROOT = Path("../../brisc2025")

SEG_TRAIN_IMAGES = DATA_ROOT / "segmentation_task" / "train" / "images"
SEG_TRAIN_MASKS = DATA_ROOT / "segmentation_task" / "train" / "masks"

SEG_TEST_IMAGES = DATA_ROOT / "segmentation_task" / "test" / "images"
SEG_TEST_MASKS = DATA_ROOT / "segmentation_task" / "test" / "masks"

OUTPUT_DIR = Path("data_preprocess")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X_TRAIN_PATH = OUTPUT_DIR / "X_train.npy"
Y_TRAIN_PATH = OUTPUT_DIR / "Y_train.npy"
X_TEST_PATH = OUTPUT_DIR / "X_test.npy"
Y_TEST_PATH = OUTPUT_DIR / "Y_test.npy"

PLANES_TRAIN_PATH = OUTPUT_DIR / "planes_train.npy"
PLANES_TEST_PATH = OUTPUT_DIR / "planes_test.npy"

IMAGE_SIZE = (512, 512)
MASK_THRESHOLD = 0.5

