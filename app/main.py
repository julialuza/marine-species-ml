from __future__ import annotations

import random
from pathlib import Path
from threading import Thread
from tkinter import filedialog

import customtkinter as ctk
import numpy as np
import tensorflow as tf
from PIL import Image, ImageSequence, ImageTk



class MarineAnimalClassifierApp(ctk.CTk):
    MODEL_PATH = Path("models/saved_model/model_efficientnetB1_13.keras")
    DATASET_TEST_DIR = Path("data/test")
    SPINNER_PATH = Path("assets/spinner.gif")

    IMG_SIZE = (240, 240)
    PREVIEW_SIZE = (320, 320)
    WINDOW_SIZE = "1000x650"
    MIN_WINDOW_SIZE = (860, 540)

    DEFAULT_THRESHOLD = 85.0

    LOADING_MESSAGES = [
        "Szukamy ryb w głębinach...",
        "Fale niosą dane...",
        "Model nurkuje...",
    ]

    COLORS = {
        "bg_main": "#0f172a",
        "preview_bg": "#1e293b",
        "text_default": "#f8fafc",
        "text_info": "#38bdf8",
        "text_warning": "#f59e0b",
        "text_error": "#ef4444",
    }

    def __init__(self) -> None:
        super().__init__()

        self.title("Rozpoznawanie zwierząt morskich")
        self.geometry(self.WINDOW_SIZE)
        self.minsize(*self.MIN_WINDOW_SIZE)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.threshold = self.DEFAULT_THRESHOLD
        self.model = self._load_model()
        self.classes = self._load_classes()

        self.last_prediction: np.ndarray | None = None
        self.gif_frames: list[ImageTk.PhotoImage] = []
        self.current_gif_frame = 0
        self.gif_running = False
        self.gif_label: ctk.CTkLabel | None = None

        self._build_ui()
        self._load_spinner()

    def _load_model(self):
        if not self.MODEL_PATH.exists():
            raise FileNotFoundError(f"Nie znaleziono modelu: {self.MODEL_PATH}")
        return tf.keras.models.load_model(self.MODEL_PATH)

    def _load_classes(self) -> list[str]:
        if not self.DATASET_TEST_DIR.exists():
            raise FileNotFoundError(f"Nie znaleziono folderu klas: {self.DATASET_TEST_DIR}")

        classes = sorted(
            item.name for item in self.DATASET_TEST_DIR.iterdir() if item.is_dir()
        )

        if not classes:
            raise ValueError("Nie znaleziono żadnych klas w dataset/test.")

        return classes

    def _build_ui(self) -> None:
        self.main_frame = ctk.CTkFrame(
            self,
            corner_radius=20,
            fg_color=self.COLORS["bg_main"],
        )
        self.main_frame.pack(expand=True, fill="both", padx=40, pady=30)

        self.select_button = ctk.CTkButton(
            self.main_frame,
            text="Wybierz zdjęcie",
            width=220,
            height=42,
            command=self.select_image,
        )
        self.select_button.pack(pady=(24, 18))

        self.result_label = ctk.CTkLabel(
            self.main_frame,
            text="Wybierz obraz do klasyfikacji",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=self.COLORS["text_default"],
        )
        self.result_label.pack(pady=(0, 14))

        self.image_label = ctk.CTkLabel(
            self.main_frame,
            text="Podgląd obrazu",
            width=self.PREVIEW_SIZE[0],
            height=self.PREVIEW_SIZE[1],
            corner_radius=16,
            fg_color=self.COLORS["preview_bg"],
        )
        self.image_label.pack(pady=10)

        self.details_button = ctk.CTkButton(
            self.main_frame,
            text="🔍 Szczegóły",
            command=self.show_top3_predictions,
            state="disabled",
            width=160,
        )
        self.details_button.pack(pady=(12, 6))
        self.details_button.pack_forget()

        self._build_threshold_section()

    def _build_threshold_section(self) -> None:
        threshold_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        threshold_frame.pack(pady=16)

        threshold_label = ctk.CTkLabel(
            threshold_frame,
            text="Próg ufności (%)",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        threshold_label.pack(side="left", padx=(0, 10))

        self.threshold_entry = ctk.CTkEntry(
            threshold_frame,
            width=90,
            justify="center",
        )
        self.threshold_entry.insert(0, str(self.threshold))
        self.threshold_entry.pack(side="left")

        apply_button = ctk.CTkButton(
            threshold_frame,
            text="Zastosuj",
            width=100,
            command=self.apply_threshold,
        )
        apply_button.pack(side="left", padx=(10, 0))

    def _load_spinner(self) -> None:
        if not self.SPINNER_PATH.exists():
            return

        spinner = Image.open(self.SPINNER_PATH)
        self.gif_frames = [
            ImageTk.PhotoImage(frame.copy().resize((128, 128)))
            for frame in ImageSequence.Iterator(spinner)
        ]

    def _start_spinner(self) -> None:
        if not self.gif_frames:
            return

        self.gif_running = True
        self.current_gif_frame = 0

        self.gif_label = ctk.CTkLabel(self.image_label, text="")
        self.gif_label.place(relx=0.5, rely=0.5, anchor="center")

        self._animate_spinner()

    def _animate_spinner(self) -> None:
        if not self.gif_running or not self.gif_label or not self.gif_frames:
            return

        frame = self.gif_frames[self.current_gif_frame]
        self.gif_label.configure(image=frame)
        self.gif_label.image = frame

        self.current_gif_frame = (self.current_gif_frame + 1) % len(self.gif_frames)
        self.after(90, self._animate_spinner)

    def _stop_spinner(self) -> None:
        self.gif_running = False
        if self.gif_label is not None:
            self.gif_label.destroy()
            self.gif_label = None

    def prepare_image(self, image_path: str) -> np.ndarray:
        image = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=self.IMG_SIZE,
        )
        image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
        return np.expand_dims(image_array, axis=0)

    def select_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Wybierz obraz",
            filetypes=[("Obrazy", "*.jpg *.jpeg *.png *.bmp *.webp")],
        )

        if not file_path:
            return

        self._set_loading_state()
        Thread(target=self._classify_in_background, args=(file_path,), daemon=True).start()

    def _set_loading_state(self) -> None:
        self.result_label.configure(
            text=random.choice(self.LOADING_MESSAGES),
            text_color=self.COLORS["text_default"],
        )
        self.image_label.configure(image=None, text="")
        self.image_label.image = None

        self.details_button.configure(state="disabled")
        self.details_button.pack_forget()

        self._start_spinner()

    def _classify_in_background(self, file_path: str) -> None:
        try:
            preview_image = Image.open(file_path).convert("RGB")
            image_array = self.prepare_image(file_path)

            prediction = self.model.predict(image_array, verbose=0)[0]
            predicted_index = int(np.argmax(prediction))
            predicted_class = self.classes[predicted_index]
            confidence = float(prediction[predicted_index] * 100)

            self.after(
                0,
                lambda: self._update_ui_with_result(
                    preview_image=preview_image,
                    prediction=prediction,
                    predicted_class=predicted_class,
                    confidence=confidence,
                ),
            )

        except Exception as error:
            self.after(0, lambda: self._handle_classification_error(error))

    def _update_ui_with_result(
        self,
        preview_image: Image.Image,
        prediction: np.ndarray,
        predicted_class: str,
        confidence: float,
    ) -> None:
        self._stop_spinner()
        self.last_prediction = prediction

        if confidence < self.threshold:
            self.result_label.configure(
                text=f"Nie rozpoznano gatunku ({confidence:.1f}%)",
                text_color=self.COLORS["text_error"],
            )
        else:
            self.result_label.configure(
                text=f"Gatunek: {predicted_class} ({confidence:.1f}%)",
                text_color=self.COLORS["text_info"],
            )

        self._show_preview(preview_image)

        self.details_button.configure(state="normal")
        self.details_button.pack(pady=(12, 6))

    def _handle_classification_error(self, error: Exception) -> None:
        self._stop_spinner()
        self.result_label.configure(
            text=f"Wystąpił błąd podczas klasyfikacji:\n{error}",
            text_color=self.COLORS["text_error"],
        )

    def _show_preview(self, image: Image.Image) -> None:
        resized = image.resize(self.PREVIEW_SIZE)
        preview = ctk.CTkImage(light_image=resized, dark_image=resized, size=self.PREVIEW_SIZE)
        self.image_label.configure(image=preview, text="")
        self.image_label.image = preview

    def show_top3_predictions(self) -> None:
        if self.last_prediction is None:
            return

        top_indices = np.argsort(self.last_prediction)[-3:][::-1]
        top_predictions = [
            (self.classes[index], float(self.last_prediction[index] * 100))
            for index in top_indices
        ]
        best_class = top_predictions[0][0]

        details_window = ctk.CTkToplevel(self)
        details_window.title("Szczegóły klasyfikacji")
        details_window.geometry("360x220")
        details_window.attributes("-topmost", True)

        title_label = ctk.CTkLabel(
            details_window,
            text="Top 3 predykcje",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        title_label.pack(pady=(18, 14))

        for class_name, confidence in top_predictions:
            weight = "bold" if class_name == best_class else "normal"
            item = ctk.CTkLabel(
                details_window,
                text=f"{class_name}: {confidence:.1f}%",
                font=ctk.CTkFont(size=15, weight=weight),
            )
            item.pack(pady=6)

    def apply_threshold(self) -> None:
        raw_value = self.threshold_entry.get().strip()

        try:
            value = float(raw_value)
        except ValueError:
            self.result_label.configure(
                text="Niepoprawna liczba",
                text_color=self.COLORS["text_error"],
            )
            return

        if not 0 <= value <= 100:
            self.result_label.configure(
                text="Wartość powinna mieścić się w zakresie 0–100",
                text_color=self.COLORS["text_error"],
            )
            return

        self.threshold = value

        message = f"Ustawiono próg ufności: {self.threshold:.1f}%"
        color = self.COLORS["text_default"]

        if value < 50:
            message += "\nUwaga: niski próg może dawać mniej wiarygodne wyniki."
            color = self.COLORS["text_warning"]

        self.result_label.configure(text=message, text_color=color)
