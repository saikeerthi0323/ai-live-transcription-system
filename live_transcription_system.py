"""
Live Transcription System  –  v3
=================================
Languages supported: English · Telugu · Hindi
UI: language selector dropdown in header

Dependencies
------------
    pip install faster-whisper sounddevice numpy
"""

import tkinter as tk
from tkinter import font as tkfont, ttk
import threading
import queue
import numpy as np

try:
    import sounddevice as sd
    from faster_whisper import WhisperModel
except ImportError as _e:
    import sys, tkinter.messagebox as mb
    root = tk.Tk(); root.withdraw()
    mb.showerror(
        "Missing dependency",
        f"{_e}\n\nPlease run:\n  pip install faster-whisper sounddevice numpy",
    )
    sys.exit(1)


# ── Colour palette ────────────────────────────────────────────────────────────
BG_DARK      = "#0d0f18"
BG_PANEL     = "#131620"
BG_PANEL_ALT = "#161926"
ACCENT_LIVE  = "#00d4aa"
ACCENT_FINAL = "#f5a623"
ACCENT_REC   = "#e05252"
TEXT_PRIMARY = "#dde1ef"
TEXT_MUTED   = "#565b72"
BORDER       = "#252840"

# ── Supported languages  (display name → Whisper language code) ───────────────
LANGUAGES = {
    "English": "en",
    "Telugu":  "te",
    "Hindi":   "hi",
}

# ── Audio constants ───────────────────────────────────────────────────────────
SAMPLE_RATE     = 16_000
CHUNK_SECS      = 0.5
CHUNK_SAMPLES   = int(SAMPLE_RATE * CHUNK_SECS)
SILENCE_THRESH  = 0.01
MAX_BUFFER_SECS = 30


class LiveTranscriptionApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Live Transcription System  v3")
        self.root.configure(bg=BG_DARK)
        self.root.minsize(960, 600)

        self.is_recording   = False
        self.transcript_buf = []
        self.stop_event     = threading.Event()
        self.q              = queue.Queue()
        self.model          = None

        # Language selection variable
        self.selected_lang = tk.StringVar(value="English")

        self._build_fonts()
        self._build_ui()
        self._poll_queue()

        threading.Thread(target=self._load_model, daemon=True).start()

    # ── Fonts ─────────────────────────────────────────────────────────────────

    def _build_fonts(self):
        self.font_title  = tkfont.Font(family="Courier New", size=11, weight="bold")
        self.font_text   = tkfont.Font(family="Consolas",    size=11)
        self.font_btn    = tkfont.Font(family="Courier New", size=10, weight="bold")
        self.font_status = tkfont.Font(family="Courier New", size=9)
        self.font_lang   = tkfont.Font(family="Courier New", size=10, weight="bold")

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header ────────────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=BG_DARK, pady=14)
        header.pack(fill="x", padx=20)

        tk.Label(header, text="⬤  LIVE TRANSCRIPTION SYSTEM  v3",
                 font=self.font_title, fg=ACCENT_LIVE, bg=BG_DARK).pack(side="left")

        self.status_lbl = tk.Label(header, text="● LOADING MODEL…",
                                   font=self.font_status, fg=ACCENT_FINAL, bg=BG_DARK)
        self.status_lbl.pack(side="right")

        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x", padx=20)

        # ── Language selector bar ─────────────────────────────────────────────
        lang_bar = tk.Frame(self.root, bg=BG_PANEL, pady=10)
        lang_bar.pack(fill="x", padx=20, pady=(0, 0))

        tk.Label(lang_bar, text="🌐  Language :",
                 font=self.font_lang, fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left", padx=(12, 8))

        # Style the radio buttons
        for lang_name in LANGUAGES:
            rb = tk.Radiobutton(
                lang_bar,
                text=lang_name,
                variable=self.selected_lang,
                value=lang_name,
                font=self.font_lang,
                fg=TEXT_PRIMARY,
                bg=BG_PANEL,
                activebackground=BG_PANEL,
                activeforeground=ACCENT_LIVE,
                selectcolor=BG_DARK,
                indicatoron=0,                  # button-style (no radio circle)
                relief="flat",
                padx=16, pady=5,
                cursor="hand2",
                command=self._on_lang_change,
            )
            rb.pack(side="left", padx=4)

        self.lang_info = tk.Label(
            lang_bar,
            text="",
            font=self.font_status,
            fg=ACCENT_LIVE, bg=BG_PANEL,
        )
        self.lang_info.pack(side="right", padx=12)

        self._refresh_lang_buttons()

        # ── Body – two panels ─────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=20, pady=16)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0)
        body.columnconfigure(2, weight=1)
        body.rowconfigure(1, weight=1)

        self._panel_header(body, col=0, icon="◈",
                           label="Live Transcription", accent=ACCENT_LIVE)
        self._panel_header(body, col=2, icon="◇",
                           label="Final Output",        accent=ACCENT_FINAL)

        self.live_text  = self._text_panel(body, row=1, col=0,
                                           accent=ACCENT_LIVE,  bg=BG_PANEL)
        tk.Frame(body, bg=BORDER, width=1).grid(
            row=0, column=1, rowspan=3, sticky="ns", padx=10)
        self.final_text = self._text_panel(body, row=1, col=2,
                                           accent=ACCENT_FINAL, bg=BG_PANEL_ALT)

        self._placeholder(self.live_text,
                          "Loading Whisper model…\nPlease wait.", TEXT_MUTED)
        self._placeholder(self.final_text,
                          "Final transcript will appear\nhere after you click STOP.",
                          TEXT_MUTED)

        # ── Footer / controls ─────────────────────────────────────────────────
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x", padx=20)
        footer = tk.Frame(self.root, bg=BG_DARK, pady=14)
        footer.pack(fill="x", padx=20)

        self.start_btn = tk.Button(
            footer, text="▶  START", font=self.font_btn,
            fg=BG_DARK, bg=TEXT_MUTED,
            activebackground="#00a88c", activeforeground=BG_DARK,
            relief="flat", padx=22, pady=8, cursor="hand2",
            state="disabled", command=self.start_recording)
        self.start_btn.pack(side="left", padx=(0, 10))

        self.stop_btn = tk.Button(
            footer, text="■  STOP", font=self.font_btn,
            fg=BG_DARK, bg=TEXT_MUTED,
            activebackground="#c04040", activeforeground=BG_DARK,
            relief="flat", padx=22, pady=8, cursor="hand2",
            state="disabled", command=self.stop_recording)
        self.stop_btn.pack(side="left", padx=(0, 30))

        self.clear_btn = tk.Button(
            footer, text="↺  CLEAR", font=self.font_btn,
            fg=TEXT_MUTED, bg=BG_PANEL,
            activebackground=BORDER, activeforeground=TEXT_PRIMARY,
            relief="flat", padx=16, pady=8, cursor="hand2",
            command=self.clear_all)
        self.clear_btn.pack(side="left")

        self.info_lbl = tk.Label(footer, text="Loading Whisper model…",
                                 font=self.font_status, fg=TEXT_MUTED, bg=BG_DARK)
        self.info_lbl.pack(side="right")

    def _panel_header(self, parent, col, icon, label, accent):
        frame = tk.Frame(parent, bg=BG_DARK)
        frame.grid(row=0, column=col, sticky="ew", pady=(0, 8))
        tk.Label(frame, text=f"{icon}  {label}",
                 font=self.font_title, fg=accent, bg=BG_DARK).pack(side="left")
        tk.Frame(frame, bg=accent, height=2).pack(side="bottom", fill="x")

    def _text_panel(self, parent, row, col, accent, bg):
        outer = tk.Frame(parent, bg=accent)
        outer.grid(row=row, column=col, sticky="nsew")
        inner = tk.Frame(outer, bg=bg, padx=2, pady=2)
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        text = tk.Text(inner, font=self.font_text,
                       fg=TEXT_PRIMARY, bg=bg,
                       insertbackground=TEXT_PRIMARY,
                       relief="flat", wrap="word",
                       padx=14, pady=12,
                       state="disabled", cursor="arrow")
        sb = tk.Scrollbar(inner, command=text.yview,
                          bg=BG_DARK, troughcolor=BG_DARK,
                          activebackground=BORDER, relief="flat")
        text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)
        return text

    def _placeholder(self, widget, msg, color):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", msg)
        widget.configure(state="disabled", fg=color)

    # ── Language helpers ──────────────────────────────────────────────────────

    def _on_lang_change(self):
        self._refresh_lang_buttons()

    def _refresh_lang_buttons(self):
        """Highlight the active language radio button."""
        lang = self.selected_lang.get()
        code = LANGUAGES[lang]
        self.lang_info.configure(
            text=f"Whisper code: [{code}]  –  speak in {lang}")

    def _get_lang_code(self):
        return LANGUAGES[self.selected_lang.get()]

    # ── Model loader ──────────────────────────────────────────────────────────

    def _load_model(self):
        try:
            self.model = WhisperModel("base", device="cpu", compute_type="int8")
            self.q.put({"type": "ready"})
        except Exception as exc:
            self.q.put({"type": "error", "text": f"Model load failed: {exc}"})

    # ── Recording control ─────────────────────────────────────────────────────

    def start_recording(self):
        if self.is_recording:
            return
        self.is_recording = True
        self.transcript_buf.clear()
        self.stop_event.clear()

        lang = self.selected_lang.get()
        self._set_text(self.live_text, "", TEXT_PRIMARY)
        self._placeholder(self.final_text,
                          "Recording in progress…\nFinal output will appear on STOP.",
                          TEXT_MUTED)
        self.start_btn.configure(state="disabled", bg=TEXT_MUTED)
        self.stop_btn.configure( state="normal",   bg=ACCENT_REC)
        self.status_lbl.configure(text="● RECORDING", fg=ACCENT_REC)
        self.info_lbl.configure(text=f"Listening in {lang}…")

        # Disable language selector while recording
        self._set_lang_selector_state("disabled")

        threading.Thread(target=self._audio_worker, daemon=True).start()

    def stop_recording(self):
        if not self.is_recording:
            return
        self.stop_event.set()
        self.is_recording = False
        self.stop_btn.configure(state="disabled", bg=TEXT_MUTED)
        self.status_lbl.configure(text="● STOPPING…", fg=ACCENT_FINAL)
        self.info_lbl.configure(text="Finalising…")

    def clear_all(self):
        if self.is_recording:
            return
        self.transcript_buf.clear()
        self._placeholder(self.live_text,
                          "Waiting to record…\nClick START to begin.", TEXT_MUTED)
        self._placeholder(self.final_text,
                          "Final transcript will appear\nhere after you click STOP.",
                          TEXT_MUTED)
        self.info_lbl.configure(text="Cleared.")

    def _set_lang_selector_state(self, state):
        """Enable / disable all radio buttons in the language bar."""
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, tk.Radiobutton):
                        child.configure(state=state)

    # ── Audio + transcription worker (background thread) ──────────────────────

    def _audio_worker(self):
        lang_code      = self._get_lang_code()   # captured at start
        audio_buffer   = []
        silence_chunks = 0
        SILENCE_LIMIT  = 2

        def _transcribe(audio_np):
            if audio_np is None or len(audio_np) == 0:
                return
            try:
                segments, _ = self.model.transcribe(
                    audio_np,
                    beam_size=5,
                    language=lang_code,         # fixed to chosen language
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=300,
                        speech_pad_ms=200,
                    ),
                )
                full = " ".join(seg.text.strip() for seg in segments).strip()
                if full:
                    self.q.put({"type": "partial", "text": full})
            except Exception as exc:
                self.q.put({"type": "error", "text": str(exc)})

        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                dtype="float32", blocksize=CHUNK_SAMPLES) as stream:
                while not self.stop_event.is_set():
                    chunk, _ = stream.read(CHUNK_SAMPLES)
                    chunk = chunk.flatten()
                    rms = float(np.sqrt(np.mean(chunk ** 2)))

                    if rms > SILENCE_THRESH:
                        audio_buffer.append(chunk)
                        silence_chunks = 0
                    else:
                        if audio_buffer:
                            silence_chunks += 1
                            audio_buffer.append(chunk)
                            buf_secs = len(audio_buffer) * CHUNK_SECS
                            if silence_chunks >= SILENCE_LIMIT or buf_secs >= MAX_BUFFER_SECS:
                                _transcribe(np.concatenate(audio_buffer))
                                audio_buffer   = []
                                silence_chunks = 0

                if audio_buffer:
                    _transcribe(np.concatenate(audio_buffer))

        except Exception as exc:
            self.q.put({"type": "error", "text": f"Audio error: {exc}"})

        self.q.put({"type": "stopped"})

    # ── Queue polling (main thread) ───────────────────────────────────────────

    def _poll_queue(self):
        try:
            while True:
                msg = self.q.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._poll_queue)

    def _handle_message(self, msg: dict):
        t = msg.get("type")

        if t == "ready":
            self.start_btn.configure(state="normal", bg=ACCENT_LIVE)
            self.status_lbl.configure(text="● IDLE", fg=TEXT_MUTED)
            self.info_lbl.configure(text="Model ready – select language & click START")
            self._placeholder(self.live_text,
                              "Waiting to record…\nClick START to begin.", TEXT_MUTED)

        elif t == "partial":
            text = msg["text"].strip()
            if text:
                self.transcript_buf.append(text)
                self._set_text(self.live_text,
                               "\n\n".join(self.transcript_buf), TEXT_PRIMARY)

        elif t == "error":
            self._append_text(self.live_text,
                              f"\n[Error: {msg['text']}]", ACCENT_REC)

        elif t == "stopped":
            final = "\n\n".join(self.transcript_buf).strip()
            if final:
                self._set_text(self.final_text, final, TEXT_PRIMARY)
            else:
                self._placeholder(self.final_text, "No speech detected.", TEXT_MUTED)
            self.start_btn.configure(state="normal",   bg=ACCENT_LIVE)
            self.stop_btn.configure( state="disabled", bg=TEXT_MUTED)
            self.status_lbl.configure(text="● IDLE", fg=TEXT_MUTED)
            self.info_lbl.configure(
                text=f"Done – {len(self.transcript_buf)} segment(s) captured.")
            self._set_lang_selector_state("normal")

    # ── Widget helpers ────────────────────────────────────────────────────────

    def _set_text(self, w: tk.Text, content: str, color: str):
        w.configure(state="normal", fg=color)
        w.delete("1.0", "end")
        w.insert("end", content)
        w.see("end")
        w.configure(state="disabled")

    def _append_text(self, w: tk.Text, content: str, color: str):
        w.configure(state="normal", fg=color)
        w.insert("end", content)
        w.see("end")
        w.configure(state="disabled")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = LiveTranscriptionApp(root)
    root.mainloop()
