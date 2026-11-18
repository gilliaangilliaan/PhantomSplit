import ctypes
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from itertools import chain, count
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import tkinter as tk
import torch
import yaml
from scipy.signal import resample_poly
from tkinter import filedialog, messagebox, ttk
from tqdm.auto import tqdm

from bs_roformer import BSRoformer

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    TkinterDnD = None

APP_VERSION = "1.0.0"
APP_ID = "gilliaan.phantomsplit"
APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "phantomsplit_config.json"
ICON_PATH = APP_DIR / "phantomsplit.ico"
LOGO_PATH = APP_DIR / "phantomsplit.png"
LINKS = (
    ("GitHub", "https://github.com/gilliaangilliaan"),
    ("Ko-fi", "https://ko-fi.com/gilliaan"),
    ("Hugging Face", "https://huggingface.co/gilliaan"),
)

MODELS_DIR = APP_DIR / "models"
MODEL_NAME = "BS-Roformer"
MODEL_URL = "https://huggingface.co/gilliaan/CenterWide-Models/resolve/main/BandSplitSpatialFormer"
MODEL_CONFIG = MODELS_DIR / "gilliaan_bsspatialroformer_centerwide_v4.yaml"
MODEL_CKPT = MODELS_DIR / "gilliaan_bsspatialroformer_centerwide_v4.ckpt"

SAMPLE_RATE = 44100
AUDIO_EXTS = (".wav", ".flac", ".m4a", ".mp3", ".aiff", ".aif")
MODES = ("mixture", "stems")
QUALITY = {"wav": ("16", "24", "32"), "flac": ("16", "24"), "mp3": ("128", "192", "256", "320")}
SUBTYPES = {"16": "PCM_16", "24": "PCM_24", "32": "FLOAT"}
DEFAULT_CONFIG = {
    "root": "", "out": "",
    "recursive": False, "input_type": "mixture", "flat": False,
    "tta": False, "batch_size": None, "num_overlap": None,
    "format": "flac", "bits": "16", "kbps": "320",
}

WORKERS = 2
GPU_LOCK = threading.Lock()
STOP = threading.Event()


class Stopped(Exception):
    pass


def check_stop():
    if STOP.is_set():
        raise Stopped


class BatchProgress:
    def __init__(self, total, emit, unit):
        self.total, self.emit, self.unit = total, emit, unit
        self.fractions, self.lock = {}, threading.Lock()

    def report(self, key, fraction):
        with self.lock:
            self.fractions[key] = fraction
            done = sum(self.fractions.values())
            position = min(sum(f >= 1 for f in self.fractions.values()) + 1, self.total)
            self.emit((round(done * 1000), self.total * 1000, f"{self.unit} {position} of {self.total}"))


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def inference_defaults():
    try:
        inference = load_yaml(MODEL_CONFIG)["inference"]
        return inference["batch_size"], inference["num_overlap"]
    except Exception:
        return 2, 2


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    try:
        saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cfg.update({k: v for k, v in saved.items() if k in cfg})
    except Exception:
        pass
    for key, valid in (("input_type", MODES), ("format", QUALITY)):
        if cfg[key] not in valid:
            cfg[key] = DEFAULT_CONFIG[key]
    for key, default in zip(("batch_size", "num_overlap"), inference_defaults()):
        cfg[key] = cfg[key] or default
    return cfg


def download_model_files(emit, on_progress):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for path in (MODEL_CONFIG, MODEL_CKPT):
        if path.exists():
            continue
        emit(f"Downloading {path.name}")
        part = path.with_name(path.name + ".part")
        with urllib.request.urlopen(f"{MODEL_URL}/{path.name}") as response, open(part, "wb") as f:
            total, done = int(response.headers.get("Content-Length") or 0), 0
            while chunk := response.read(1 << 20):
                check_stop()
                f.write(chunk)
                done += len(chunk)
                if total:
                    on_progress(done, total)
        part.replace(path)


def load_engine(batch_size, num_overlap, tta):
    config = load_yaml(MODEL_CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BSRoformer(**{**config["model"], "use_torch_checkpoint": False})
    model.load_state_dict(torch.load(str(MODEL_CKPT), map_location="cpu", weights_only=True))
    return {
        "name": MODEL_NAME + (" (TTA)" if tta else ""), "model": model.to(device).eval(), "device": device,
        "chunk_size": config["audio"]["chunk_size"], "instruments": config["training"]["instruments"],
        "batch_size": batch_size, "num_overlap": num_overlap, "tta": tta,
    }


def load_audio(path):
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if sr == SAMPLE_RATE:
        return audio.T
    g = gcd(sr, SAMPLE_RATE)
    return np.stack([resample_poly(ch, SAMPLE_RATE // g, sr // g) for ch in audio.T]).astype("float32")


def write_audio(path, audio, quality):
    fmt = path.suffix[1:]
    if quality != "32":
        audio = np.clip(audio, -1.0, 1.0)
    if fmt == "mp3":
        options = {
            "format": "MP3", "subtype": "MPEG_LAYER_III", "bitrate_mode": "CONSTANT",
            "compression_level": (320 - int(quality)) / 288,
        }
    else:
        options = {"subtype": SUBTYPES[quality]}
        if fmt == "flac":
            options["compression_level"] = 1.0
    sf.write(str(path), audio.T, SAMPLE_RATE, **options)


def demix(engine, mix, report):
    c, batch_size = engine["chunk_size"], engine["batch_size"]
    step = c // engine["num_overlap"]
    border = c - step
    padded = mix.shape[-1] > 2 * border > 0
    if padded:
        mix = np.pad(mix, ((0, 0), (border, border)), mode="reflect")

    fade = max(c // 10, 1)
    window = np.ones(c, dtype=np.float32)
    window[:fade] = np.linspace(0, 1, fade, dtype=np.float32)
    window[-fade:] = np.linspace(1, 0, fade, dtype=np.float32)

    total = mix.shape[-1]
    result = np.zeros((len(engine["instruments"]), 2, total), dtype=np.float32)
    counter = np.zeros((1, total), dtype=np.float32)
    positions = range(0, max(total - c, 0) + step, step)

    bar = tqdm(total=len(positions), desc=engine["name"], unit="chunk", leave=False, disable=sys.stderr is None)
    with GPU_LOCK, torch.no_grad(), bar:
        for start in range(0, len(positions), batch_size):
            check_stop()
            batch = positions[start:start + batch_size]
            chunks = [mix[:, pos:pos + c] for pos in batch]
            stacked = np.stack([np.pad(chunk, ((0, 0), (0, c - chunk.shape[-1]))) for chunk in chunks])
            out = engine["model"](torch.tensor(stacked, dtype=torch.float32, device=engine["device"])).cpu().numpy()
            for pos, chunk, separated in zip(batch, chunks, out):
                n = chunk.shape[-1]
                result[..., pos:pos + n] += separated[..., :n] * window[:n]
                counter[..., pos:pos + n] += window[:n]
            bar.update(len(batch))
            report((start + len(batch)) / len(positions))

    result /= np.maximum(counter, 1e-8)
    if padded:
        result = result[..., border:-border]
    stems = dict(zip(engine["instruments"], result))
    return stems["center"], stems["wide"]


def mono_fold(audio):
    return np.repeat(audio.mean(axis=0, keepdims=True), len(audio), axis=0)


def split(audio, engine, report):
    if len(audio) == 1:
        center = np.repeat(audio, 2, axis=0)
        return center, np.zeros_like(center)

    if engine["tta"]:
        center, wide = demix(engine, audio, lambda fraction: report(fraction / 2))
        center2, wide2 = demix(engine, audio[:, ::-1].copy(), lambda fraction: report((1 + fraction) / 2))
        center, wide = (center + center2[:, ::-1]) / 2, (wide + wide2[:, ::-1]) / 2
    else:
        center, wide = demix(engine, audio, report)
    return mono_fold(center), audio - mono_fold(audio - wide)


def sum_mix(tracks):
    out = np.zeros((tracks[0].shape[0], max(t.shape[1] for t in tracks)), dtype=np.float32)
    for track in tracks:
        out[:, :track.shape[1]] += track
    return out


def sanitize_name(name):
    return re.sub(r"[^0-9a-z\-_]", "", name.lower())


def find_audio(folder, recursive=False):
    files = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in files if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def unique_flat_name(clean_name, taken):
    numbered = (f"{clean_name}{i}" for i in count(2))
    name = next(n for n in chain([clean_name], numbered) if n not in taken)
    taken.add(name)
    return name


def resolve_out_dir(out_root, clean_name, source_name):
    out_dir = out_root / clean_name
    marker = out_dir / ".source_name"
    if not marker.exists() or marker.read_text(encoding="utf-8").strip() == source_name:
        return out_dir
    return next(d for i in count(2) if not (d := out_root / f"{clean_name}{i}").exists())


def plan_tasks(sources, out, flat, fmt, emit):
    tasks, taken = [], set()
    for i, source in enumerate(sources, 1):
        name = sanitize_name(source.stem if source.is_file() else source.name) or f"track{i}"
        if flat:
            name = unique_flat_name(name, taken)
            folder, prefix, label = out, f"{name}_", f"{name}_*"
        else:
            folder = resolve_out_dir(out, name, source.name)
            prefix, label = "", folder.name
        paths = [folder / f"{prefix}{track}.{fmt}" for track in ("center", "wide")]
        header = f"[{i}/{len(sources)}] {source.name} -> {label}"
        if all(path.exists() for path in paths):
            emit(f"{header} already done, skipping")
            continue
        if not flat:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / ".source_name").write_text(source.name, encoding="utf-8")
        tasks.append((source, paths, header))
    return tasks


def run_batch(root, out, engine, emit, *, recursive, stems_mode, flat, fmt, quality):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    note = engine["name"]
    if stems_mode:
        stems = find_audio(root)
        sources = [root] if stems else []
        note = f"mixed from {len(stems)} stems, {note}"
        if not stems:
            emit(f"  [warn] No stem files found in {root}")
    else:
        sources = [root] if root.is_file() else find_audio(root, recursive)
        emit(f"Found {len(sources)} mixture file(s)")

    tasks = plan_tasks(sources, out, flat, fmt, emit)
    progress = BatchProgress(len(stems) if stems_mode else len(tasks), emit, "Stem" if stems_mode else "Song")

    def split_file(path):
        try:
            audio = load_audio(path)
            if len(audio) == 1:
                emit("    (Mono, passed through to center)")
            return split(audio, engine, lambda fraction: progress.report(path, fraction))
        finally:
            progress.report(path, 1)

    def split_stems():
        def split_stem(path):
            emit(f"    Stem: {path.name}")
            return split_file(path)

        with ThreadPoolExecutor(WORKERS) as executor:
            return [sum_mix(tracks) for tracks in zip(*executor.map(split_stem, stems))]

    def run_task(source, paths, header):
        if STOP.is_set():
            return
        emit(header)
        try:
            tracks = split_stems() if stems_mode else split_file(source)
            for path, audio in zip(paths, tracks):
                write_audio(path, audio, quality)
            emit(f"  Wrote {' / '.join(p.name for p in paths)} ({note}) -> {paths[0].parent}")
        except Stopped:
            pass
        except Exception as e:
            emit(f"  [error] Failed on {source.name}: {e}")

    if len(tasks) > 1:
        emit(f"Processing {len(tasks)} file(s), {WORKERS} at a time")
    with ThreadPoolExecutor(WORKERS) as executor:
        list(executor.map(lambda task: run_task(*task), tasks))
    emit("Stopped" if STOP.is_set() else "Done")


def gpu_info():
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)}"
    return "GPU: none detected (CPU only)"


def open_folder(path):
    if not Path(path).exists():
        return
    if os.name == "nt":
        subprocess.Popen(["explorer", str(Path(path))])
    else:
        webbrowser.open(Path(path).as_uri())


def format_left(seconds):
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d} left" if hours else f"{minutes}:{secs:02d} left"


BG = "#0d0d0f"
CARD_BG = "#161619"
CARD_BORDER = "#242428"
FG = "#e9e9ec"
FG_MUTED = "#87878f"
ACCENT = "#ff4fa3"
ACCENT_DIM = "#d6318a"
ACCENT_FG = BG
INPUT_BG = "#1b1b1f"
INPUT_BORDER = "#2c2c31"
INPUT_HOVER = "#242429"
DOT_OFF = "#4a4a52"
FONT = "Segoe UI"


def spaced(text):
    return " ".join(text)


def toggle_colors(active, inactive_fg=FG):
    return (ACCENT, ACCENT_FG, ACCENT_DIM) if active else (INPUT_BG, inactive_fg, INPUT_HOVER)


def highlight(buttons, selected):
    for name, btn in buttons.items():
        bg, fg, active = toggle_colors(name == selected, inactive_fg=FG_MUTED)
        btn.configure(bg=bg, fg=fg, activebackground=active)


def enable_dark_titlebar(win):
    if os.name != "nt":
        return
    try:
        win.update()
        hwnd = ctypes.windll.user32.GetParent(win.winfo_id())
        value = ctypes.c_int(1)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(value), ctypes.sizeof(value))
    except Exception:
        pass


def put(widget, **pack):
    widget.pack(**pack)
    return widget


def frame(parent, **kw):
    kw.setdefault("bg", parent.cget("bg"))
    return tk.Frame(parent, **kw)


def label(parent, text, muted=False, size=9, bold=False, **kw):
    kw.setdefault("bg", parent.cget("bg"))
    kw.setdefault("fg", FG_MUTED if muted else FG)
    kw.setdefault("font", (FONT, size, "bold") if bold else (FONT, size))
    return tk.Label(parent, text=text, **kw)


def button(parent, text, command, accent=False, small=False, **kw):
    bg, fg, active = toggle_colors(accent)
    kw.setdefault("padx", 14)
    return tk.Button(
        parent, text=text, command=command, bg=bg, fg=fg,
        activebackground=active, activeforeground=fg, relief="flat", bd=0,
        font=(FONT, 9 if small else 10, "bold"), pady=6 if small else 8,
        cursor="hand2", highlightthickness=0, **kw,
    )


def entry(parent, var, **kw):
    return tk.Entry(
        parent, textvariable=var, bg=INPUT_BG, fg=FG, insertbackground=FG,
        disabledbackground=INPUT_BG, disabledforeground=FG_MUTED,
        relief="flat", highlightthickness=1, highlightbackground=INPUT_BORDER,
        highlightcolor=ACCENT, font=(FONT, 10), **kw,
    )


def field(parent, text, var):
    put(label(parent, text), side="left")
    return put(entry(parent, var, width=6), side="left", padx=(6, 24), ipady=2)


def card(parent, title, subtitle, **pack):
    outer = put(
        tk.Frame(parent, bg=CARD_BG, highlightbackground=CARD_BORDER, highlightthickness=1),
        **{"fill": "x", "padx": 18, "pady": 8, **pack},
    )
    header = put(frame(outer), fill="x", padx=16, pady=(12, 8))
    put(frame(header, bg=ACCENT, width=3), side="left", fill="y", padx=(0, 10))
    text_col = put(frame(header), side="left", fill="x", expand=True)
    put(label(text_col, spaced(title.upper()), size=10, bold=True), anchor="w")
    put(label(text_col, subtitle, muted=True, size=8), anchor="w")
    return put(frame(outer), fill="both", expand=True, padx=16, pady=(0, 14))


def path_field(body, text, var, actions):
    row = put(frame(body), fill="x", pady=(0, 8))
    put(label(row, text, width=6, anchor="w"), side="left")
    box = put(entry(row, var), side="left", fill="x", expand=True, ipady=4)
    buttons = [
        put(button(row, name, command, small=True, width=12, padx=4), side="left", padx=(8, 0))
        for name, command in actions
    ]
    return box, buttons


class Toggle(tk.Frame):
    def __init__(self, parent, text, var):
        super().__init__(parent, bg=parent.cget("bg"), cursor="hand2")
        self.var, self.enabled = var, True
        self.switch = put(tk.Canvas(self, width=34, height=18, bg=self.cget("bg"), highlightthickness=0), side="left")
        self.text = put(label(self, text), side="left", padx=(8, 0))
        for widget in (self, self.switch, self.text):
            widget.bind("<Button-1>", self.toggle)
        var.trace_add("write", lambda *_: self.draw())
        self.draw()

    def toggle(self, event=None):
        if self.enabled:
            self.var.set(not self.var.get())

    def configure(self, state=None, **kw):
        if state is not None:
            self.enabled = state != "disabled"
            self.draw()
        if kw:
            super().configure(**kw)

    def draw(self):
        on = self.var.get()
        track = CARD_BORDER if not self.enabled else ACCENT if on else INPUT_BORDER
        knob = FG_MUTED if not self.enabled else ACCENT_FG if on else FG
        x = 20 if on else 4
        c = self.switch
        c.delete("all")
        c.create_oval(1, 1, 17, 17, fill=track, outline=track)
        c.create_oval(17, 1, 33, 17, fill=track, outline=track)
        c.create_rectangle(9, 1, 25, 17, fill=track, outline=track)
        c.create_oval(x, 4, x + 10, 14, fill=knob, outline=knob)
        self.text.configure(fg=FG if self.enabled else FG_MUTED)


class BatchGUI(tk.Tk, *((TkinterDnD.DnDWrapper,) if TkinterDnD else ())):
    def __init__(self):
        super().__init__()
        self.title(f"PhantomSplit {APP_VERSION}")
        if ICON_PATH.exists():
            self.iconbitmap(default=str(ICON_PATH))
        self.geometry("820x800")
        self.resizable(False, False)
        self.configure(bg=BG)
        enable_dark_titlebar(self)
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "Accent.Horizontal.TProgressbar", troughcolor=CARD_BG, background=ACCENT,
            bordercolor=CARD_BG, lightcolor=ACCENT, darkcolor=ACCENT, thickness=6,
        )

        cfg = load_config()
        self.vars = {
            key: (tk.BooleanVar if isinstance(value, bool) else tk.StringVar)(value=value)
            for key, value in cfg.items()
        }
        self.log_queue = queue.Queue()
        self.worker = None

        self.build_footer()
        self.build_header()
        self.build_mode_and_status()
        self.build_inference()
        self.build_paths()
        self.set_input_type(cfg["input_type"])
        self.set_format(cfg["format"])
        self.set_running(False)
        self.update_disk_label()
        self.vars["out"].trace_add("write", self.update_disk_label)
        self.enable_drop()
        self.after(100, self.poll_log_queue)

    def build_footer(self):
        footer = put(
            frame(self, highlightbackground=CARD_BORDER, highlightthickness=1),
            side="bottom", fill="x", padx=18, pady=14,
        )
        put(button(footer, "Save config", self.on_save, small=True), side="left", padx=14, pady=12)
        for text, url in LINKS:
            link = put(label(footer, text, muted=True, cursor="hand2"), side="left", padx=(0, 14))
            link.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))
            link.bind("<Enter>", lambda e: e.widget.configure(fg=ACCENT))
            link.bind("<Leave>", lambda e: e.widget.configure(fg=FG_MUTED))
        self.run_btn = put(button(footer, spaced("PROCESS"), self.on_run, accent=True), side="right", padx=14, pady=12)
        self.stop_btn = put(button(footer, spaced("STOP"), self.on_stop), side="right", pady=12)

    def build_header(self):
        topbar = put(frame(self), fill="x", padx=18, pady=(16, 6))
        if LOGO_PATH.exists():
            self.logo = tk.PhotoImage(file=str(LOGO_PATH))
            put(tk.Label(topbar, image=self.logo, bg=BG, bd=0), side="right", padx=(0, 2))
        heading = put(frame(topbar), side="left")
        wordmark = put(frame(heading), anchor="w")
        for text, kw in (("PHANTOM", {}), (" • ", {"fg": ACCENT}), ("SPLIT", {"muted": True})):
            put(label(wordmark, text, size=13, bold=True, **kw), side="left")
        put(label(heading, f"v{APP_VERSION}  •  {gpu_info()}", muted=True, size=8), anchor="w", padx=(4, 0))

    def build_mode_and_status(self):
        top = put(frame(self), fill="x", padx=18, pady=8)
        halves = []
        for column, padx in enumerate(((0, 8), (8, 0))):
            top.columnconfigure(column, weight=1, uniform="top")
            halves.append(frame(top))
            halves[-1].grid(row=0, column=column, sticky="nsew", padx=padx)
        side_by_side = {"fill": "both", "expand": True, "padx": 0, "pady": 0}

        mode = card(halves[0], "process mode", "Mixture files, or one song's stems to mix down", **side_by_side)
        row = put(frame(mode), anchor="w")
        self.mode_buttons = {
            name: put(button(row, name.capitalize(), lambda n=name: self.set_input_type(n)), side="left", padx=(0, 8))
            for name in MODES
        }
        self.recursive_check = put(
            Toggle(mode, "Recursive (search subfolders)", self.vars["recursive"]), anchor="w", pady=(10, 0),
        )

        status = card(halves[1], "status", "Current batch state", **side_by_side)
        row = put(frame(status), fill="x")
        self.status_dot = put(tk.Canvas(row, width=10, height=10, bg=CARD_BG, highlightthickness=0), side="left")
        self.status_oval = self.status_dot.create_oval(1, 1, 9, 9, outline="")
        self.status_label = put(
            label(row, "Ready", muted=True, anchor="w", width=1), side="left", padx=(6, 0), fill="x", expand=True,
        )
        self.progress = put(
            ttk.Progressbar(status, mode="determinate", style="Accent.Horizontal.TProgressbar"),
            fill="x", pady=(12, 6),
        )
        self.progress_label = put(label(status, "", muted=True, anchor="w", width=1), fill="x")

    def build_inference(self):
        v = self.vars
        inference = card(
            self, "inference", f"{MODEL_NAME}: more overlap is slower, a bigger batch needs more GPU memory",
        )
        row = put(frame(inference), anchor="w")
        for key, text in (("batch_size", "Batch size"), ("num_overlap", "Overlap")):
            field(row, text, v[key])

        row = put(frame(inference), anchor="w", pady=(10, 0))
        put(label(row, "Format"), side="left", padx=(0, 6))
        self.format_buttons = {
            name: put(button(row, name.upper(), lambda n=name: self.set_format(n), small=True), side="left", padx=(0, 6))
            for name in QUALITY
        }
        self.quality_menu = put(tk.OptionMenu(row, v["bits"], ""), side="left", padx=(8, 4))
        self.quality_menu.configure(
            bg=INPUT_BG, fg=FG, activebackground=INPUT_HOVER, activeforeground=FG, relief="flat", bd=0,
            highlightthickness=0, font=(FONT, 10), width=4, cursor="hand2", indicatoron=False,
        )
        self.quality_menu["menu"].configure(
            bg=INPUT_BG, fg=FG, activebackground=ACCENT, activeforeground=ACCENT_FG, bd=0, font=(FONT, 10), tearoff=0,
        )
        self.quality_unit = put(label(row, ""), side="left")
        put(
            Toggle(inference, "TTA (also split the song played backwards and average, 2x processing time)", v["tta"]),
            anchor="w", pady=(10, 0),
        )

    def build_paths(self):
        v = self.vars
        paths = card(
            self, "paths",
            "Input: a file or folder (in stems mode, one song's stems)  •  Output: a folder  •  Drag and drop works",
        )
        self.browse_file_btn = path_field(paths, "Input", v["root"], [
            ("Browse file", lambda: self.browse("root", folder=False)),
            ("Browse folder", lambda: self.browse("root")),
        ])[1][0]
        self.out_entry = path_field(paths, "Output", v["out"], [
            ("Browse", lambda: self.browse("out")),
            ("Open folder", lambda: open_folder(v["out"].get())),
        ])[0]
        self.disk_label = put(label(paths, "", muted=True, size=8), anchor="w", pady=(2, 0))
        put(
            Toggle(
                paths, "Flat mode (songname_center / songname_wide all in one folder, no per-song folders)", v["flat"],
            ),
            anchor="w", pady=(6, 0),
        )

    def enable_drop(self):
        if TkinterDnD is None:
            return
        try:
            TkinterDnD._require(self)
        except Exception:
            return
        for widget, key in ((self, "root"), (self.out_entry, "out")):
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", lambda e, k=key: self.on_drop(k, e.data))

    def on_drop(self, key, data):
        dropped = [Path(item) for item in self.tk.splitlist(data)]
        if not dropped:
            return
        path = dropped[0]
        if path.is_file() and (key == "out" or len(dropped) > 1 or self.vars["input_type"].get() == "stems"):
            path = path.parent
        self.vars[key].set(path.as_posix())

    def browse(self, key, folder=True):
        if folder:
            path = filedialog.askdirectory()
        else:
            patterns = " ".join(f"*{ext}" for ext in AUDIO_EXTS)
            path = filedialog.askopenfilename(filetypes=[("audio files", patterns), ("all files", "*.*")])
        if path:
            self.vars[key].set(path)

    def values(self):
        return {key: var.get() for key, var in self.vars.items()}

    def quality_key(self):
        return "kbps" if self.vars["format"].get() == "mp3" else "bits"

    def inference_settings(self):
        batch_size, num_overlap = int(self.vars["batch_size"].get()), int(self.vars["num_overlap"].get())
        if batch_size < 1 or num_overlap < 1:
            raise ValueError
        return batch_size, num_overlap

    def set_input_type(self, name):
        self.vars["input_type"].set(name)
        highlight(self.mode_buttons, name)
        state = "disabled" if name == "stems" else "normal"
        self.browse_file_btn.configure(state=state)
        self.recursive_check.configure(state=state)

    def set_format(self, name):
        self.vars["format"].set(name)
        highlight(self.format_buttons, name)
        key, choices = self.quality_key(), QUALITY[name]
        var = self.vars[key]
        if var.get() not in choices:
            var.set(choices[-1])
        menu = self.quality_menu["menu"]
        menu.delete(0, "end")
        for choice in choices:
            menu.add_command(label=choice, command=lambda c=choice: var.set(c))
        self.quality_menu.configure(textvariable=var)
        self.quality_unit.configure(text=key)

    def set_running(self, running):
        self.run_btn.configure(state="disabled" if running else "normal")
        self.stop_btn.configure(state="normal" if running else "disabled")
        self.status_dot.itemconfig(self.status_oval, fill=ACCENT if running else DOT_OFF)
        self.progress.configure(value=0)
        self.progress_label.configure(text="")
        self.progress_mark = None

    def update_disk_label(self, *_):
        try:
            text = f"Free space: {shutil.disk_usage(self.vars['out'].get()).free / 1024 ** 3:.1f} GB"
        except (OSError, ValueError):
            text = ""
        self.disk_label.configure(text=text)

    def show_progress(self, done, total, position=None):
        now = time.monotonic()
        if not self.progress_mark or done < self.progress_mark[2]:
            self.progress_mark = (now, done, done)
        started, first, _ = self.progress_mark
        self.progress_mark = (started, first, done)
        text = "  •  ".join(filter(None, (position, f"{100 * done // total}%")))
        if first < done < total:
            text += "  " + format_left(round((now - started) * (total - done) / (done - first)))
        self.progress.configure(maximum=total, value=done)
        self.progress_label.configure(text=text)

    def poll_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                if msg is None:
                    self.set_running(False)
                elif isinstance(msg, tuple):
                    self.show_progress(*msg)
                else:
                    print(msg)
                    self.status_label.configure(text=msg)
        except queue.Empty:
            pass
        self.after(100, self.poll_log_queue)

    def validate(self):
        v = self.values()
        if not v["root"] or not Path(v["root"]).exists():
            return "Input file or folder does not exist"
        if v["input_type"] == "stems" and not Path(v["root"]).is_dir():
            return "Stems mode requires a folder of stem files"
        if not v["out"]:
            return "Output folder is required"
        try:
            self.inference_settings()
        except ValueError:
            return "Batch size and overlap must be whole numbers, 1 or more"
        return None

    def on_save(self):
        CONFIG_PATH.write_text(json.dumps(self.values(), indent=2), encoding="utf-8")

    def on_stop(self):
        STOP.set()
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="Stopping")

    def on_run(self):
        if self.worker and self.worker.is_alive():
            return
        if error := self.validate():
            messagebox.showerror("PhantomSplit", error)
            return

        self.on_save()
        v, settings, quality = self.values(), self.inference_settings(), self.vars[self.quality_key()].get()
        STOP.clear()
        self.set_running(True)
        self.status_label.configure(text="Running")
        emit = self.log_queue.put

        def work():
            try:
                download_model_files(emit, lambda done, total: emit((done, total)))
                emit(f"Loading {MODEL_NAME}")
                run_batch(
                    v["root"], v["out"], load_engine(*settings, v["tta"]), emit,
                    recursive=v["recursive"], stems_mode=v["input_type"] == "stems", flat=v["flat"],
                    fmt=v["format"], quality=quality,
                )
            except Stopped:
                emit("Stopped")
            except Exception as e:
                print(traceback.format_exc())
                emit(f"[error] {e}")
            emit(None)

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()


if __name__ == "__main__":
    if os.name == "nt":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    BatchGUI().mainloop()
