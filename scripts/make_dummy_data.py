"""Create tiny synthetic data to test the pipeline before using real MRI data."""
from pathlib import Path
import numpy as np


def make_blob(h=256, w=256):
    img = np.random.normal(0, 0.05, (h, w)).astype(np.float32)
    mask = np.zeros((h, w), dtype=np.float32)
    cy = np.random.randint(h // 4, 3 * h // 4)
    cx = np.random.randint(w // 4, 3 * w // 4)
    ry = np.random.randint(10, 35)
    rx = np.random.randint(10, 35)
    yy, xx = np.ogrid[:h, :w]
    ellipse = ((yy - cy) ** 2 / (ry ** 2) + (xx - cx) ** 2 / (rx ** 2)) <= 1
    mask[ellipse] = 1.0
    img += mask * np.random.uniform(0.6, 1.2)
    img += np.random.normal(0, 0.02, (h, w)).astype(np.float32)
    return img, mask


def main():
    out = Path("data")
    out.mkdir(exist_ok=True)
    X_train, Y_train = zip(*[make_blob() for _ in range(64)])
    X_test, Y_test = zip(*[make_blob() for _ in range(16)])
    np.save(out / "X_train.npy", np.stack(X_train))
    np.save(out / "Y_train.npy", np.stack(Y_train))
    np.save(out / "X_test.npy", np.stack(X_test))
    np.save(out / "Y_test.npy", np.stack(Y_test))
    print("Saved dummy data to ./data")


if __name__ == "__main__":
    main()
