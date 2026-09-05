# ./visualize_predictions.py
"""
Randomly sample exactly five examples from a CSV dataset, run the trained model
on them, and draw the 28x28 images so the model quality can be confirmed visually.

The loader handles both CSV layouts used in this project automatically:
  * label,pixel0,...,pixel783   (e.g. data/train.csv)  -> true labels available
  * pixel0,...,pixel783         (e.g. data/test.csv)   -> no labels, visual check only

When true labels exist the figure titles show "pred / true" (green = correct,
red = wrong) and the full-set accuracy is printed; otherwise each image is
labelled with the predicted class and its softmax confidence.

Always samples exactly five examples. The draw is truly random on every run;
pass --seed to reproduce a particular draw.

Usage:
    python visualize_predictions.py
    python visualize_predictions.py --data ./data/test.csv --model ./model_params.npz \
                                    --seed 42 --output ./test_preview.png
    python visualize_predictions.py --show        # also pop up a matplotlib window
"""

import argparse
import csv
import os
import sys

import numpy as np

from model.Net import Net


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample test examples, predict with the trained model, and "
                    "visualize the results for manual confirmation."
    )
    parser.add_argument("--data", default="./data/test.csv",
                        help="CSV file with images (default: %(default)s)")
    parser.add_argument("--model", default="./model_params.npz",
                        help="Model parameters exported by Net.save / train.py "
                             "(default: %(default)s)")
    parser.add_argument("--seed", type=int, default=None,
                        help="optional seed to reproduce a fixed random draw; "
                             "omit it for a fresh random sample every run")
    parser.add_argument("--output", default="./test_preview.png",
                        help="path of the saved preview figure (default: %(default)s)")
    parser.add_argument("--show", action="store_true",
                        help="also open an interactive matplotlib window")
    return parser.parse_args()


def _is_float(token):
    try:
        float(token)
        return True
    except (TypeError, ValueError):
        return False


def load_samples(path, input_dim):
    """
    Load images (and labels when present) from a CSV file.

    Returns:
        X: float array of shape (N, input_dim), raw 0-255 pixel values
        y: int array of shape (N,) if the file contains a label column, else None
    """
    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError(f"File is empty: {path}")

    # Drop a header line (its first cell is not numeric, e.g. "label"/"pixel0").
    if not _is_float(rows[0][0]):
        rows = rows[1:]
    if not rows:
        raise ValueError(f"File contains no data rows: {path}")

    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError(f"Rows have inconsistent column counts: {sorted(widths)}")
    width = widths.pop()

    if width == input_dim:            # features only (e.g. data/test.csv)
        offset, has_labels = 0, False
    elif width == input_dim + 1:      # label column + features (e.g. data/train.csv)
        offset, has_labels = 1, True
    else:
        raise ValueError(
            f"Expected {input_dim} or {input_dim + 1} columns per row, "
            f"but the file has {width}"
        )

    X = np.array([[float(value) for value in row[offset:]] for row in rows],
                 dtype=np.float64)
    y = np.array([int(row[0]) for row in rows], dtype=np.int64) if has_labels else None
    return X, y


def predict(net, X):
    """Run a full forward pass. X must already be preprocessed like training data."""
    X_norm = X / 255.0  # same normalization as train.py
    logits = net.forward(X_norm)

    # Softmax probabilities (per row) to report a confidence per sample.
    z = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(z)
    probs /= probs.sum(axis=1, keepdims=True)

    return logits.argmax(axis=1), probs.max(axis=1)


def plot_samples(images, titles, title_colors, suptitle, output_path, show):
    """Draw the sampled images in a grid and save/show the figure."""
    import matplotlib.pyplot as plt

    num = len(images)
    cols = 5
    rows = (num + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")

    for i, (img, title, color) in enumerate(zip(images, titles, title_colors)):
        ax = axes[i]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=11, color=color)

    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"Preview figure saved to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        sys.exit(
            f"Model file not found: {args.model}\n"
            "Run `python train.py` first to train the network and export "
            "model_params.npz (see README.md)."
        )

    net = Net.load(args.model)
    input_dim = net.layers[0].input_dim
    side = int(round(input_dim ** 0.5))
    if side * side != input_dim:
        sys.exit(f"Input dimension {input_dim} is not a perfect square; "
                 "cannot draw 2D images.")

    X, y = load_samples(args.data, input_dim)
    n_total = X.shape[0]
    if n_total == 0:
        sys.exit("No samples could be loaded from the data file.")
    if y is not None:
        print(f"Loaded {n_total} samples of shape {X.shape} (with labels)")
    else:
        print(f"Loaded {n_total} samples of shape {X.shape} "
              "(no label column -> visual confirmation only)")

    preds, confs = predict(net, X)

    if y is not None:
        accuracy = float(np.mean(preds == y))
        print(f"Full-set accuracy on '{os.path.basename(args.data)}': {accuracy:.4f}")

    num_samples = 5  # fixed number of samples to draw and display
    num = min(num_samples, n_total)
    if n_total < num_samples:
        print(f"Note: dataset has only {n_total} samples (< {num_samples}); "
              "showing all of them.")
    rng = np.random.default_rng(args.seed)  # None -> fresh random draw per run
    indices = rng.choice(n_total, size=num, replace=False)

    images, titles, title_colors = [], [], []
    print("\nSampled predictions:")
    for j, i in enumerate(indices):
        pred, conf = int(preds[i]), float(confs[i])
        if y is not None:
            true = int(y[i])
            correct = pred == true
            mark = "OK" if correct else "WRONG"
            print(f"  [{j}] sample #{i}: true={true}, predicted={pred} "
                  f"(confidence={conf:.3f}) -> {mark}")
            titles.append(f"pred {pred} / true {true} ({conf:.2f})")
            title_colors.append("green" if correct else "red")
        else:
            print(f"  [{j}] sample #{i}: predicted={pred} (confidence={conf:.3f})")
            titles.append(f"pred {pred} ({conf:.2f})")
            title_colors.append("black")
        images.append(X[i].reshape(side, side))

    suptitle = (f"Model preview on random samples from "
                f"{os.path.basename(args.data)}")
    plot_samples(images, titles, title_colors, suptitle, args.output, args.show)

    if y is None:
        print("\nNote: this data file has no true labels - please check by eye "
              "that each digit image matches its predicted class.")


if __name__ == "__main__":
    main()
