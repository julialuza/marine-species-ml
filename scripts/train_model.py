from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf


TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")

MODEL_SAVE_PATH = Path("models/saved_model/model_efficientnetB1_13.keras")
REPORTS_DIR = Path("reports/figures")

IMG_SIZE = (240, 240)
BATCH_SIZE = 32
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 30

FROZEN_LAYERS = 80
INITIAL_LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 1e-5
DROPOUT_RATE = 0.3
DENSE_UNITS = 384
LABEL_SMOOTHING = 0.05
EARLY_STOPPING_PATIENCE = 7


def get_class_names(train_dir: Path) -> list[str]:
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    classes = sorted(item.name for item in train_dir.iterdir() if item.is_dir())

    if not classes:
        raise ValueError("No class folders found in training directory.")

    return classes


def create_data_generators():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )

    return train_generator, val_generator


def build_model(num_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    base_model = tf.keras.applications.EfficientNetB1(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Dense(DENSE_UNITS, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    return model, base_model


def freeze_base_layers(base_model: tf.keras.Model, frozen_layers: int) -> None:
    for layer in base_model.layers[:frozen_layers]:
        layer.trainable = False

    for layer in base_model.layers[frozen_layers:]:
        layer.trainable = True


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=LABEL_SMOOTHING
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=["accuracy"],
    )


def plot_training_history(history_phase_1, history_phase_2, output_path: Path) -> None:
    train_acc = history_phase_1.history["accuracy"] + history_phase_2.history["accuracy"]
    val_acc = history_phase_1.history["val_accuracy"] + history_phase_2.history["val_accuracy"]
    train_loss = history_phase_1.history["loss"] + history_phase_2.history["loss"]
    val_loss = history_phase_1.history["val_loss"] + history_phase_2.history["val_loss"]

    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Model Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def train_model() -> None:
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    class_names = get_class_names(TRAIN_DIR)
    train_generator, val_generator = create_data_generators()

    model, base_model = build_model(num_classes=len(class_names))

    freeze_base_layers(base_model, FROZEN_LAYERS)
    compile_model(model, INITIAL_LEARNING_RATE)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    print("\nPHASE 1: training classifier head")
    history_phase_1 = model.fit(
        train_generator,
        epochs=INITIAL_EPOCHS,
        validation_data=val_generator,
    )

    print("\nPHASE 2: fine-tuning")
    for layer in base_model.layers:
        layer.trainable = True

    compile_model(model, FINE_TUNE_LEARNING_RATE)

    history_phase_2 = model.fit(
        train_generator,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=val_generator,
        callbacks=[early_stopping],
    )

    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")

    plot_training_history(
        history_phase_1=history_phase_1,
        history_phase_2=history_phase_2,
        output_path=REPORTS_DIR / "accuracy_plot.png",
    )


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()