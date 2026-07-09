"""Create synthetic multi-lesion data to test component-aware lesion crop quickly."""
from pathlib import Path
import numpy as np


def make_multilesion_blob(h=256, w=256, max_components=4):
    img = np.random.normal(0, 0.05, (h, w)).astype(np.float32)
    mask = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.ogrid[:h, :w]

    n_comp = np.random.randint(1, max_components + 1)
    for _ in range(n_comp):
        cy = np.random.randint(h // 8, 7 * h // 8)
        cx = np.random.randint(w // 8, 7 * w // 8)
        # Mix small and medium lesions. Small ones mimic the missed components.
        if np.random.rand() < 0.65:
            ry = np.random.randint(4, 12)
            rx = np.random.randint(4, 12)
        else:
            ry = np.random.randint(12, 30)
            rx = np.random.randint(12, 30)
        ellipse = ((yy - cy) ** 2 / (ry ** 2) + (xx - cx) ** 2 / (rx ** 2)) <= 1
        mask[ellipse] = 1.0
        img += ellipse.astype(np.float32) * np.random.uniform(0.45, 1.1)

    # Add some low-contrast confusing regions.
    for _ in range(np.random.randint(0, 3)):
        cy = np.random.randint(h // 8, 7 * h // 8)
        cx = np.random.randint(w // 8, 7 * w // 8)
        ry = np.random.randint(8, 25)
        rx = np.random.randint(8, 25)
        ellipse = ((yy - cy) ** 2 / (ry ** 2) + (xx - cx) ** 2 / (rx ** 2)) <= 1
        img += ellipse.astype(np.float32) * np.random.uniform(0.08, 0.20)

    img += np.random.normal(0, 0.02, (h, w)).astype(np.float32)
    return img.astype(np.float32), mask.astype(np.float32)


def main():
    out = Path("data_multilesion")
    out.mkdir(exist_ok=True)
    X_train, Y_train = zip(*[make_multilesion_blob() for _ in range(96)])
    X_test, Y_test = zip(*[make_multilesion_blob() for _ in range(24)])
    np.save(out / "X_train.npy", np.stack(X_train))
    np.save(out / "Y_train.npy", np.stack(Y_train))
    np.save(out / "X_test.npy", np.stack(X_test))
    np.save(out / "Y_test.npy", np.stack(Y_test))
    print("Saved synthetic multi-lesion data to ./data_multilesion")


if __name__ == "__main__":
    main()
