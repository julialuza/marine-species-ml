from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = Path("models/saved_model/model_efficientnetB1_13.keras")
TEST_DIR = Path("data/test")
REPORTS_DIR = Path("reports/figures")

IMG_SIZE = (240, 240)
BATCH_SIZE = 32


def load_test_generator(test_dir: Path, img_size: tuple[int, int], batch_size: int):
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return test_generator


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
    output_path: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(np.arange(len(class_labels)), class_labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(class_labels)), class_labels)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_precision_per_class(
    report: dict,
    class_labels: list[str],
    output_path: Path,
) -> None:
    precisions = [report[label]["precision"] for label in class_labels]

    plt.figure(figsize=(14, 6))
    bars = plt.bar(class_labels, precisions)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Precision")
    plt.title("Precision per Class")

    for bar, value in zip(bars, precisions):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.2f}",
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def evaluate_model() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(MODEL_PATH)
    test_generator = load_test_generator(TEST_DIR, IMG_SIZE, BATCH_SIZE)

    class_labels = list(test_generator.class_indices.keys())
    y_true = test_generator.classes

    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_labels,
        output_dict=True,
    )

    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_labels=class_labels,
        output_path=REPORTS_DIR / "confusion_matrix.png",
    )

    plot_precision_per_class(
        report=report,
        class_labels=class_labels,
        output_path=REPORTS_DIR / "precision_per_class.png",
    )

    accuracy = report["accuracy"]
    macro_avg = report["macro avg"]
    weighted_avg = report["weighted avg"]

    print("\nEVALUATION SUMMARY")
    print(f"Accuracy:      {accuracy:.4f}")
    print(f"Macro Precision:   {macro_avg['precision']:.4f}")
    print(f"Macro Recall:      {macro_avg['recall']:.4f}")
    print(f"Macro F1-score:    {macro_avg['f1-score']:.4f}")
    print(f"Weighted Precision:{weighted_avg['precision']:.4f}")
    print(f"Weighted Recall:   {weighted_avg['recall']:.4f}")
    print(f"Weighted F1-score: {weighted_avg['f1-score']:.4f}")

def main() -> None:
    evaluate_model()

if __name__ == "__main__":
    main()