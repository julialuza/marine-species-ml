from __future__ import annotations

import random
import tkinter as tk
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

    PREVIEW_SIZE = (700, 700)
    SPINNER_SIZE = (96, 96)

    WINDOW_SIZE = "1000x700"
    MIN_WINDOW_SIZE = (1000, 700)

    DEFAULT_THRESHOLD = 85.0

    LOADING_MESSAGES = [
        "Szukamy ryb w głębinach...",
        "Fale niosą dane...",
        "Model nurkuje..."
    ]

    COLORS = {
        "bg": "#080f1a",
        "surface": "#0d1b2e",
        "border": "#162840",
        "border_active": "#00c9b1",
        "teal": "#00c9b1",
        "teal_dim": "#00796b",
        "text": "#dceef7",
        "text_muted": "#4a7a96",
        "error": "#ff4d6d",
        "warning": "#f59e0b",
    }

    def __init__(self) -> None:
        super().__init__()

        self.title("Klasyfikator gatunków morskich")
        self.geometry(self.WINDOW_SIZE)
        self.minsize(*self.MIN_WINDOW_SIZE)
        self.configure(fg_color=self.COLORS["bg"])

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.threshold = self.DEFAULT_THRESHOLD
        self.model = self._load_model()
        self.classes = self._load_classes()

        self.last_prediction: np.ndarray | None = None
        self.preview_image_ref: ImageTk.PhotoImage | None = None

        self.spinner_frames: list[ImageTk.PhotoImage] = []
        self.current_spinner_frame = 0
        self.spinner_running = False
        self.spinner_after_id: str | None = None
        self.spinner_label: tk.Label | None = None

        self.is_classifying = False

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
            raise ValueError("Nie znaleziono żadnych klas w data/test.")
        return classes

    def _build_ui(self) -> None:
        wrap = ctk.CTkFrame(self, fg_color="transparent")
        wrap.pack(expand=True, fill="both", padx=36, pady=28)

        # title
        tk.Label(
            wrap,
            text="Klasyfikator gatunków morskich",
            font=("Georgia", 40, "bold"),
            fg=self.COLORS["text"],
            bg=self.COLORS["bg"],
        ).pack(anchor="center")

        tk.Label(
            wrap,
            text="Klasyfikacja zdjęć · EfficientNetB1",
            font=("Helvetica Neue", 15),
            fg=self.COLORS["text_muted"],
            bg=self.COLORS["bg"],
        ).pack(anchor="center", pady=(6, 30))

        # upload button
        self.select_button = ctk.CTkButton(
            wrap,
            text="Wybierz zdjęcie",
            width=230,
            height=48,
            corner_radius=10,
            font=ctk.CTkFont(family="Helvetica Neue", size=15, weight="bold"),
            fg_color=self.COLORS["teal"],
            hover_color="#00a896",
            text_color=self.COLORS["bg"],
            command=self.select_image,
        )
        self.select_button.pack(anchor="center", pady=(0, 26))

        # result label
        self.result_label = ctk.CTkLabel(
            wrap,
            text="Wybierz obraz do klasyfikacji",
            font=ctk.CTkFont(family="Georgia", size=22, weight="bold"),
            text_color=self.COLORS["text_muted"],
            anchor="center",
            justify="center",
        )
        self.result_label.pack(fill="x", pady=(0, 16))

        # preview
        self.preview_frame = tk.Frame(
            wrap,
            width=self.PREVIEW_SIZE[0],
            height=self.PREVIEW_SIZE[1],
            bg=self.COLORS["surface"],
            highlightthickness=1,
            highlightbackground=self.COLORS["border"],
        )
        self.preview_frame.pack(anchor="center", pady=(0, 0))
        self.preview_frame.pack_propagate(False)

        self.preview_placeholder = tk.Label(
            self.preview_frame,
            text="Podgląd obrazu",
            font=("Helvetica Neue", 16),
            fg=self.COLORS["text_muted"],
            bg=self.COLORS["surface"],
        )
        self.preview_placeholder.place(relx=0.5, rely=0.5, anchor="center")

        self.preview_image_label = tk.Label(
            self.preview_frame,
            bd=0,
            highlightthickness=0,
            bg=self.COLORS["surface"],
        )
        self.preview_image_label.place(relx=0.5, rely=0.5, anchor="center")

        # confidence bar - zostaje niewyśrodkowany / pełna szerokość
        self.conf_bar_canvas = tk.Canvas(
            wrap,
            height=6,
            bg=self.COLORS["bg"],
            highlightthickness=0,
        )
        self.conf_bar_canvas.pack(fill="x", pady=(10, 20))
        self._conf_pct = 0.0
        self.conf_bar_canvas.bind("<Configure>", lambda e: self._redraw_conf_bar())

        # details button
        self.details_button = ctk.CTkButton(
            wrap,
            text="🔍 Szczegóły predykcji",
            width=220,
            height=42,
            corner_radius=10,
            font=ctk.CTkFont(family="Helvetica Neue", size=13),
            fg_color="transparent",
            hover_color=self.COLORS["surface"],
            text_color=self.COLORS["teal"],
            border_width=1,
            border_color=self.COLORS["teal_dim"],
            command=self.show_top3_predictions,
            state="disabled",
        )
        self.details_button.pack(anchor="center", pady=(6, 24))
        self.details_button.pack_forget()

        # divider
        tk.Frame(wrap, height=1, bg=self.COLORS["border"]).pack(fill="x", pady=(0, 22))

        # threshold
        self._build_threshold_section(wrap)

    def _build_threshold_section(self, parent) -> None:
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(anchor="center")

        row = ctk.CTkFrame(section, fg_color="transparent")
        row.pack(anchor="center")

        tk.Label(
            row,
            text="Próg ufności",
            font=("Helvetica Neue", 13),
            fg=self.COLORS["text_muted"],
            bg=self.COLORS["bg"],
        ).pack(side="left", padx=(0, 12))

        self.threshold_entry = ctk.CTkEntry(
            row,
            width=72,
            height=36,
            corner_radius=8,
            justify="center",
            font=ctk.CTkFont(family="Courier New", size=13),
            fg_color=self.COLORS["surface"],
            border_color=self.COLORS["border"],
            text_color=self.COLORS["teal"],
        )
        self.threshold_entry.insert(0, str(int(self.threshold)))
        self.threshold_entry.pack(side="left")

        tk.Label(
            row,
            text="%",
            font=("Helvetica Neue", 13),
            fg=self.COLORS["text_muted"],
            bg=self.COLORS["bg"],
        ).pack(side="left", padx=6)

        ctk.CTkButton(
            row,
            text="Zastosuj",
            width=96,
            height=36,
            corner_radius=8,
            font=ctk.CTkFont(family="Helvetica Neue", size=12),
            fg_color=self.COLORS["surface"],
            hover_color=self.COLORS["border"],
            text_color=self.COLORS["text"],
            border_width=1,
            border_color=self.COLORS["border"],
            command=self.apply_threshold,
        ).pack(side="left", padx=(8, 0))

    def _redraw_conf_bar(self) -> None:
        c = self.conf_bar_canvas
        w = c.winfo_width()
        h = c.winfo_height()
        c.delete("all")
        c.create_rectangle(0, 0, w, h, fill=self.COLORS["border"], outline="")
        fill_w = int(w * self._conf_pct / 100)
        if fill_w > 0:
            color = (
                self.COLORS["teal"]
                if self._conf_pct >= 80
                else self.COLORS["warning"]
                if self._conf_pct >= 50
                else self.COLORS["error"]
            )
            c.create_rectangle(0, 0, fill_w, h, fill=color, outline="")

    def _set_conf_bar(self, pct: float) -> None:
        self._conf_pct = pct
        self._redraw_conf_bar()

    def _load_spinner(self) -> None:
        if not self.SPINNER_PATH.exists():
            return
        spinner = Image.open(self.SPINNER_PATH)
        self.spinner_frames = [
            ImageTk.PhotoImage(
                frame.copy().convert("RGBA").resize(self.SPINNER_SIZE)
            )
            for frame in ImageSequence.Iterator(spinner)
        ]

    def _start_spinner(self) -> None:
        if not self.spinner_frames:
            return
        self._stop_spinner()
        self.spinner_running = True
        self.current_spinner_frame = 0
        self.spinner_label = tk.Label(
            self.preview_frame,
            bd=0,
            highlightthickness=0,
            bg=self.COLORS["surface"],
        )
        self.spinner_label.place(relx=0.5, rely=0.5, anchor="center")
        self._animate_spinner()

    def _animate_spinner(self) -> None:
        if not self.spinner_running or not self.spinner_frames:
            return
        frame = self.spinner_frames[self.current_spinner_frame]
        self.spinner_label.configure(image=frame)
        self.spinner_label.image = frame
        self.current_spinner_frame = (self.current_spinner_frame + 1) % len(self.spinner_frames)
        self.spinner_after_id = self.after(90, self._animate_spinner)

    def _stop_spinner(self) -> None:
        self.spinner_running = False
        if self.spinner_after_id is not None:
            try:
                self.after_cancel(self.spinner_after_id)
            except ValueError:
                pass
            self.spinner_after_id = None
        if self.spinner_label is not None:
            self.spinner_label.destroy()
            self.spinner_label = None

    def _clear_preview(self) -> None:
        self.preview_image_ref = None
        self.preview_image_label.configure(image="")
        self.preview_image_label.image = None
        self.preview_placeholder.place_forget()
        self._set_conf_bar(0)
        self.preview_frame.configure(highlightbackground=self.COLORS["border"])

    def prepare_image(self, image_path: str) -> np.ndarray:
        image = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=self.IMG_SIZE,
        )
        arr = tf.keras.preprocessing.image.img_to_array(image) / 255.0
        return np.expand_dims(arr, axis=0)

    def select_image(self) -> None:
        if self.is_classifying:
            return
        file_path = filedialog.askopenfilename(
            title="Wybierz obraz",
            filetypes=[("Obrazy", "*.jpg *.jpeg *.png *.bmp *.webp")],
        )
        if not file_path:
            return
        self._set_loading_state()
        Thread(target=self._classify_in_background, args=(file_path,), daemon=True).start()

    def _set_loading_state(self) -> None:
        self.is_classifying = True
        self.select_button.configure(state="disabled")
        self.result_label.configure(
            text=random.choice(self.LOADING_MESSAGES),
            text_color=self.COLORS["text_muted"],
        )
        self._clear_preview()
        self.details_button.configure(state="disabled")
        self.details_button.pack_forget()
        self._start_spinner()
        self.preview_frame.configure(highlightbackground=self.COLORS["teal_dim"])

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
                text=f"Nierozpoznany gatunek ({confidence:.1f}%)",
                text_color=self.COLORS["error"],
            )
        else:
            self.result_label.configure(
                text=f"{predicted_class} — {confidence:.1f}%",
                text_color=self.COLORS["teal"],
            )

        self._set_conf_bar(confidence)
        self._show_preview(preview_image)

        self.details_button.configure(state="normal")
        self.details_button.pack(anchor="center", pady=(8, 24))

        self.is_classifying = False
        self.select_button.configure(state="normal")

    def _handle_classification_error(self, error: Exception) -> None:
        self._stop_spinner()
        self._clear_preview()
        self.result_label.configure(
            text=f"Błąd klasyfikacji: {error}",
            text_color=self.COLORS["error"],
        )
        self.is_classifying = False
        self.select_button.configure(state="normal")

    def _show_preview(self, image: Image.Image) -> None:
        self.preview_placeholder.place_forget()
        img = image.copy()
        ratio = img.width / img.height
        fw, fh = self.PREVIEW_SIZE

        if ratio > fw / fh:
            nw, nh = fw, int(fw / ratio)
        else:
            nh, nw = fh, int(fh * ratio)

        resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        self.preview_image_ref = photo
        self.preview_image_label.configure(image=photo)
        self.preview_image_label.image = photo
        self.preview_frame.configure(highlightbackground=self.COLORS["border_active"])

    def show_top3_predictions(self) -> None:
        if self.last_prediction is None:
            return

        top_indices = np.argsort(self.last_prediction)[-3:][::-1]
        top_predictions = [
            (self.classes[i], float(self.last_prediction[i] * 100))
            for i in top_indices
        ]

        win = ctk.CTkToplevel(self)
        win.title("Szczegóły klasyfikacji")
        win.geometry("460x360")  # trochę większe okno
        win.configure(fg_color=self.COLORS["bg"])
        win.attributes("-topmost", True)
        win.resizable(False, False)

        tk.Label(
            win,
            text="Top 3 predykcje",
            font=("Georgia", 22, "bold"),  # większy tytuł
            fg=self.COLORS["text"],
            bg=self.COLORS["bg"],
        ).pack(pady=(28, 6))

        tk.Label(
            win,
            text="Rozkład pewności modelu",
            font=("Helvetica Neue", 13),  # większy opis
            fg=self.COLORS["text_muted"],
            bg=self.COLORS["bg"],
        ).pack(pady=(0, 22))

        medals = ["🥇", "🥈", "🥉"]

        for idx, (class_name, confidence) in enumerate(top_predictions):
            row = tk.Frame(
                win,
                bg=self.COLORS["surface"],
                highlightthickness=1,
                highlightbackground=self.COLORS["border_active"] if idx == 0 else self.COLORS["border"],
            )
            row.pack(fill="x", padx=32, pady=6)

            inner = tk.Frame(row, bg=self.COLORS["surface"])
            inner.pack(fill="x", padx=16, pady=14)  # większy padding

            tk.Label(
                inner,
                text=f"{medals[idx]}  {class_name}",
                font=("Helvetica Neue", 16, "bold" if idx == 0 else "normal"),  # większy tekst
                fg=self.COLORS["teal"] if idx == 0 else self.COLORS["text_muted"],
                bg=self.COLORS["surface"],
                anchor="w",
            ).pack(side="left")

            tk.Label(
                inner,
                text=f"{confidence:.1f}%",
                font=("Courier New", 15, "bold"),  # większe %
                fg=self.COLORS["teal"] if idx == 0 else self.COLORS["text_muted"],
                bg=self.COLORS["surface"],
            ).pack(side="right")

    def apply_threshold(self) -> None:
        raw = self.threshold_entry.get().strip()
        try:
            value = float(raw)
        except ValueError:
            self.result_label.configure(
                text="Niepoprawna wartość progu",
                text_color=self.COLORS["error"],
            )
            return

        if not 0 <= value <= 100:
            self.result_label.configure(
                text="Wartość powinna być w zakresie 0–100",
                text_color=self.COLORS["error"],
            )
            return

        self.threshold = value
        text = f"Próg ustawiony: {self.threshold:.0f}%"
        color = self.COLORS["text"]
        if value < 50:
            text += "  ⚠ Niski próg"
            color = self.COLORS["warning"]

        self.result_label.configure(text=text, text_color=color)
