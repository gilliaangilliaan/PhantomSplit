import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from math import gcd
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import yaml
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tqdm.auto import tqdm

from scipy.signal import resample_poly

from bs_roformer import BSRoformer

try:
    import pyflac
    HAVE_PYFLAC = True
except ImportError:
    HAVE_PYFLAC = False

SAMPLE_RATE = 44100
AUDIO_EXTS = (".wav", ".flac", ".m4a", ".mp3", ".aiff", ".aif")

MODELS_DIR = Path(__file__).resolve().parent / "models"
APP_VERSION = "1.0.0"
MODEL_NAME = "BSRoformer"
MODEL_URL = "https://huggingface.co/gilliaan/CenterWide-Models/resolve/main/BandSplitSpatialFormer"
MODEL_CONFIG = MODELS_DIR / "gilliaan_bsspatialroformer_centerwide_v4.yaml"
MODEL_CKPT = MODELS_DIR / "gilliaan_bsspatialroformer_centerwide_v4.ckpt"
PARALLEL_SPLITS = 2
GPU_LOCK = threading.Lock()

CONFIG_PATH = Path(__file__).resolve().parent / "phantomsplit_config.json"
ICON_PATH = Path(__file__).resolve().parent / "phantomsplit.ico"
APP_ID = "gilliaan.phantomsplit"
LINKS = (
    ("github", "https://github.com/gilliaangilliaan"),
    ("ko-fi", "https://ko-fi.com/gilliaan"),
    ("hugging face", "https://huggingface.co/gilliaan"),
)
DEFAULT_CONFIG = {
    "root": "", "out": "",
    "recursive": False, "input_type": "mixture", "flat": False,
    "tta": False, "batch_size": None, "num_overlap": None,
}


def sanitize_name(name):
    return re.sub(r"[^0-9a-z\-_]", "", name.lower())


def resample(audio, sr, target_sr):
    if sr == target_sr:
        return audio
    g = gcd(sr, target_sr)
    up, down = target_sr // g, sr // g
    return np.stack([resample_poly(ch, up, down) for ch in audio], axis=0).astype("float32")


class Cfg(dict):
    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Cfg(value) if isinstance(value, dict) else value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class YamlLoader(yaml.FullLoader):
    pass


YamlLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def load_yaml_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return Cfg(yaml.load(f, Loader=YamlLoader))


def prefer_target_instrument(config):
    if config.training.target_instrument:
        return [config.training.target_instrument]
    return list(config.training.instruments)


def load_audio(path, target_sr):
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    audio = resample(audio.T, sr, target_sr)
    return audio, audio.shape[0] == 1


def write_flac(path, audio, sr):
    audio = resample(audio, sr, SAMPLE_RATE)
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    if HAVE_PYFLAC:
        tmp_fd, tmp_name = tempfile.mkstemp(suffix=".wav", dir=str(path.parent))
        os.close(tmp_fd)
        tmp = Path(tmp_name)
        sf.write(str(tmp), pcm16.T, SAMPLE_RATE, subtype="PCM_16")
        try:
            pyflac.FileEncoder(input_file=tmp, output_file=path, compression_level=8).process()
            return
        except Exception as e:
            print(f"    [warn] pyflac failed ({e}), using soundfile instead")
        finally:
            tmp.unlink(missing_ok=True)
    sf.write(str(path), pcm16.T, SAMPLE_RATE, subtype="PCM_16")


def find_state_dict(obj, _seen=None):
    if _seen is None:
        _seen = set()
    if id(obj) in _seen or not isinstance(obj, dict) or not obj:
        return None
    _seen.add(id(obj))
    if all(torch.is_tensor(v) for v in obj.values()):
        return obj
    for v in obj.values():
        found = find_state_dict(v, _seen)
        if found is not None:
            return found
    return None


def extract_state_dict(ckpt):
    found = find_state_dict(ckpt)
    if found is not None:
        return found
    if isinstance(ckpt, dict):
        tensor_only = {k: v for k, v in ckpt.items() if torch.is_tensor(v)}
        if tensor_only:
            return tensor_only
        kinds = {k: type(v).__name__ for k, v in ckpt.items()}
        raise RuntimeError(f"could not find a tensor state dict inside checkpoint; top-level keys: {kinds}")
    return ckpt


def download_model_files(emit, on_progress=None):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for path in (MODEL_CONFIG, MODEL_CKPT):
        if path.exists():
            continue
        emit(f"downloading {path.name}")
        part = path.with_name(path.name + ".part")
        with urllib.request.urlopen(f"{MODEL_URL}/{path.name}") as response, open(part, "wb") as f:
            total, done = int(response.headers.get("Content-Length") or 0), 0
            while chunk := response.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if on_progress and total:
                    on_progress(done, total)
        part.replace(path)


def load_engine(batch_size, num_overlap, tta=False):
    for path in (MODEL_CONFIG, MODEL_CKPT):
        if not path.exists():
            raise FileNotFoundError(f"{path.name} is missing from {MODELS_DIR}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_yaml_config(MODEL_CONFIG)
    model = BSRoformer(**{**config.model, "use_torch_checkpoint": False})
    model.load_state_dict(extract_state_dict(torch.load(str(MODEL_CKPT), map_location="cpu", weights_only=True)))
    model.to(device).eval()
    return {
        "name": MODEL_NAME + (" (tta)" if tta else ""), "model": model, "config": config, "device": device,
        "batch_size": batch_size, "num_overlap": num_overlap, "tta": tta,
    }


def load_engines(batch_size, num_overlap, tta=False, count=PARALLEL_SPLITS):
    return [load_engine(batch_size, num_overlap, tta)] * count


def model_inference_defaults():
    try:
        inference = load_yaml_config(MODEL_CONFIG).inference
        return inference.batch_size, inference.num_overlap
    except Exception:
        return 2, 2


class EnginePool:
    def __init__(self, engines):
        self.size = len(engines)
        self.engine = engines[0]
        self.free = queue.Queue()
        for e in engines:
            self.free.put(e)

    def run(self, fn, *args, **kwargs):
        engine = self.free.get()
        try:
            return fn(*args, engine=engine, **kwargs)
        finally:
            self.free.put(engine)


def demix_chunked(model, config, mix, device, batch_size, num_overlap, desc=None, on_progress=None):
    c = config.audio.chunk_size
    step = c // num_overlap
    fade_size = max(c // 10, 1)
    border = c - step

    length_init = mix.shape[-1]
    if length_init > 2 * border and border > 0:
        mix = np.pad(mix, ((0, 0), (border, border)), mode="reflect")

    window = np.ones(c, dtype=np.float32)
    window[:fade_size] = np.linspace(0, 1, fade_size, dtype=np.float32)
    window[-fade_size:] = np.linspace(1, 0, fade_size, dtype=np.float32)

    instruments = prefer_target_instrument(config)
    total_length = mix.shape[-1]
    result = np.zeros((len(instruments), 2, total_length), dtype=np.float32)
    counter = np.zeros((1, total_length), dtype=np.float32)

    positions = []
    i = 0
    while i < total_length:
        positions.append(i)
        if i + c >= total_length:
            break
        i += step

    progress = tqdm(total=len(positions), desc=desc or "Processing audio chunks", unit="chunk", leave=False, disable=sys.stderr is None)
    with torch.no_grad(), progress:
        for b_start in range(0, len(positions), batch_size):
            batch_positions = positions[b_start:b_start + batch_size]
            batch_chunks = []
            lens = []
            for pos in batch_positions:
                chunk = mix[:, pos:pos + c]
                lens.append(chunk.shape[-1])
                if chunk.shape[-1] < c:
                    chunk = np.pad(chunk, ((0, 0), (0, c - chunk.shape[-1])))
                batch_chunks.append(chunk)
            batch_tensor = torch.tensor(np.stack(batch_chunks), dtype=torch.float32, device=device)
            out = model(batch_tensor).cpu().numpy()
            for j, pos in enumerate(batch_positions):
                seg_len = lens[j]
                w = window[:seg_len]
                result[:, :, pos:pos + seg_len] += out[j, :, :, :seg_len] * w
                counter[:, pos:pos + seg_len] += w
            progress.update(len(batch_positions))
            if on_progress:
                on_progress(min(b_start + batch_size, len(positions)), len(positions))

    result /= np.maximum(counter, 1e-8)[None, ...]

    if length_init > 2 * border and border > 0:
        result = result[:, :, border:-border]

    return {inst: result[idx] for idx, inst in enumerate(instruments)}


def run_engine(engine, audio, part=0, parts=1):
    on_progress = engine.get("on_progress")
    scaled = (lambda done, total: on_progress(part * total + done, parts * total)) if on_progress else None
    with GPU_LOCK:
        out = demix_chunked(
            engine["model"], engine["config"], audio, engine["device"],
            engine["batch_size"], engine["num_overlap"], desc=engine["name"],
            on_progress=scaled,
        )
    return out["center"], out["wide"]


def extract_center_wide(audio, is_mono, engine):
    if is_mono:
        center = np.vstack([audio[0], audio[0]])
        return center, np.zeros_like(center)

    parts = 2 if engine.get("tta") else 1
    center, wide = run_engine(engine, audio, 0, parts)
    if parts == 2:
        center2, wide2 = run_engine(engine, audio[:, ::-1].copy(), 1, parts)
        center, wide = (center + center2[:, ::-1]) / 2, (wide + wide2[:, ::-1]) / 2

    inverted_center = audio - wide
    wide = audio - mono_fold(inverted_center)

    return mono_fold(center), wide


def mono_fold(audio):
    mono = audio.mean(axis=0, keepdims=True)
    return np.repeat(mono, audio.shape[0], axis=0)


def sum_mix(chunks):
    n = max(c.shape[1] for c in chunks)
    out = np.zeros((chunks[0].shape[0], n), dtype=np.float32)
    for c in chunks:
        out[:, :c.shape[1]] += c
    return out


def output_names(file_prefix=""):
    return tuple(f"{file_prefix}{name}.flac" for name in ("center", "wide"))


def write_output(out_dir, center, wide, engine, emit, prefix="", file_prefix=""):
    out_dir.mkdir(parents=True, exist_ok=True)
    names = output_names(file_prefix)
    for name, audio in zip(names, (center, wide)):
        write_flac(out_dir / name, audio, SAMPLE_RATE)
    emit(f"  wrote {' / '.join(names)} ({prefix}{engine['name']}) -> {out_dir}")


def process_file(src_path, engine, out_dir, emit, file_prefix=""):
    audio, is_mono = load_audio(src_path, SAMPLE_RATE)
    if is_mono:
        emit("    (mono, passthrough to center)")
    center, wide = extract_center_wide(audio, is_mono, engine)
    write_output(out_dir, center, wide, engine, emit, file_prefix=file_prefix)


def split_stem(stem_path, emit, engine):
    emit(f"    stem: {stem_path.name}")
    audio, is_mono = load_audio(stem_path, SAMPLE_RATE)
    return extract_center_wide(audio, is_mono, engine)


def process_stems_folder(folder, pool, out_dir, emit, file_prefix=""):
    stem_files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if not stem_files:
        emit(f"  [warn] no stem files found in {folder}")
        return

    with ThreadPoolExecutor(max_workers=pool.size) as executor:
        results = list(executor.map(lambda p: pool.run(split_stem, p, emit), stem_files))
    centers = [center for center, _ in results]
    wides = [wide for _, wide in results]

    write_output(
        out_dir, sum_mix(centers), sum_mix(wides), pool.engine, emit,
        prefix=f"mixed from {len(stem_files)} stems, ", file_prefix=file_prefix,
    )


def is_complete(out_dir, file_prefix=""):
    return all((out_dir / name).exists() for name in output_names(file_prefix))


def unique_flat_name(clean_name, taken):
    name, n = clean_name, 2
    while name in taken:
        name = f"{clean_name}{n}"
        n += 1
    taken.add(name)
    return name


def resolve_out_dir(out_root, clean_name, source_name):
    out_dir = out_root / clean_name
    if not out_dir.exists():
        return out_dir
    marker = out_dir / ".source_name"
    existing = marker.read_text(encoding="utf-8").strip() if marker.exists() else None
    if existing is None or existing == source_name:
        return out_dir
    n = 2
    while (out_root / f"{clean_name}{n}").exists():
        n += 1
    return out_root / f"{clean_name}{n}"


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            saved = {}
        cfg.update({k: v for k, v in saved.items() if k in DEFAULT_CONFIG})
    if cfg["input_type"] not in ("mixture", "stems"):
        cfg["input_type"] = "mixture"
    for key, default in zip(("batch_size", "num_overlap"), model_inference_defaults()):
        cfg[key] = cfg[key] or default
    return cfg


def save_config(values):
    CONFIG_PATH.write_text(json.dumps(values, indent=2), encoding="utf-8")


def disk_free_gb(path):
    try:
        return shutil.disk_usage(path).free / (1024 ** 3)
    except Exception:
        return None


def gpu_info():
    if torch.cuda.is_available():
        return f"gpu: {torch.cuda.get_device_name(0)}"
    return "gpu: none detected (cpu only)"


def open_folder(path_str):
    p = Path(path_str)
    if not p.exists():
        return
    if os.name == "nt":
        subprocess.Popen(["explorer", str(p)])
    else:
        webbrowser.open(p.as_uri())


def run_batch(root_str, out_str, recursive, engines, emit, stems_mode=False, flat=False):
    pool = EnginePool(engines)
    root = Path(root_str) if root_str else None
    if not root or not root.exists():
        emit(f"[error] input path does not exist: {root_str}")
        return
    if not out_str:
        emit("[error] output folder is required")
        return
    out = Path(out_str)

    if stems_mode:
        if not root.is_dir():
            emit("[error] stems mode requires a folder of stem files as input")
            return
        clean_name = sanitize_name(root.name) or "track1"
        if flat:
            out_dir, file_prefix, label = out, f"{clean_name}_", f"{clean_name}_*"
        else:
            out_dir = resolve_out_dir(out, clean_name, root.name)
            file_prefix, label = "", out_dir.name
        if is_complete(out_dir, file_prefix):
            emit(f"{label} already done, skipping")
            return
        emit(f"mixing down stems in {root.name} -> {label}")
        try:
            process_stems_folder(root, pool, out_dir, emit, file_prefix=file_prefix)
        except Exception as e:
            emit(f"[error] failed on {root.name}: {e}")
            return
        if not flat:
            (out_dir / ".source_name").write_text(root.name, encoding="utf-8")
        emit("done")
        return

    out.mkdir(parents=True, exist_ok=True)
    if not flat:
        complete = {d.name for d in out.iterdir() if d.is_dir() and is_complete(d)}
        emit(f"output folder already has {len(complete)} complete tracks - these will be skipped")

    if root.is_file():
        src_files = [root]
    else:
        finder = root.rglob("*") if recursive else root.iterdir()
        src_files = sorted(p for p in finder if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    emit(f"found {len(src_files)} mixture file(s)")

    tasks = []
    taken = set()
    for i, src_path in enumerate(src_files, 1):
        clean_name = sanitize_name(src_path.stem) or f"track{i}"
        if flat:
            name = unique_flat_name(clean_name, taken)
            out_dir, file_prefix, label = out, f"{name}_", f"{name}_*"
            done = is_complete(out, file_prefix)
        else:
            out_dir = resolve_out_dir(out, clean_name, src_path.name)
            file_prefix, label = "", out_dir.name
            done = out_dir.name in complete
        if done:
            emit(f"[{i}/{len(src_files)}] {src_path.name} -> {label} already done, skipping")
            continue
        if not flat:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / ".source_name").write_text(src_path.name, encoding="utf-8")
        tasks.append((i, src_path, out_dir, file_prefix, label))

    def run_task(i, src_path, out_dir, file_prefix, label):
        emit(f"[{i}/{len(src_files)}] {src_path.name} -> {label}")
        try:
            pool.run(
                process_file, src_path, out_dir=out_dir, emit=emit,
                file_prefix=file_prefix,
            )
        except Exception as e:
            emit(f"  [error] failed on {src_path.name}: {e}")

    if len(tasks) > 1:
        emit(f"processing {len(tasks)} file(s), {pool.size} at a time")
    with ThreadPoolExecutor(max_workers=pool.size) as executor:
        for future in [executor.submit(run_task, *task) for task in tasks]:
            future.result()
    emit("done")


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


def enable_windows_dark_titlebar(win):
    if os.name != "nt":
        return
    try:
        import ctypes
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


def button(parent, text, command, accent=False, small=False):
    bg, fg, active = toggle_colors(accent)
    return tk.Button(
        parent, text=text, command=command, bg=bg, fg=fg,
        activebackground=active, activeforeground=fg, relief="flat", bd=0,
        font=(FONT, 9 if small else 10, "bold"), padx=14, pady=6 if small else 8,
        cursor="hand2", highlightthickness=0,
    )


def entry(parent, var, **kw):
    return tk.Entry(
        parent, textvariable=var, bg=INPUT_BG, fg=FG, insertbackground=FG,
        disabledbackground=INPUT_BG, disabledforeground=FG_MUTED,
        relief="flat", highlightthickness=1, highlightbackground=INPUT_BORDER,
        highlightcolor=ACCENT, font=(FONT, 10), **kw,
    )


def check(parent, text, var):
    bg = parent.cget("bg")
    return tk.Checkbutton(
        parent, text=text, variable=var, bg=bg, fg=FG,
        activebackground=bg, activeforeground=FG, selectcolor=INPUT_BG,
        relief="flat", bd=0, highlightthickness=0, font=(FONT, 9), cursor="hand2",
    )


def field(parent, text, var, suffix=""):
    put(label(parent, text), side="left")
    box = put(entry(parent, var, width=6), side="left", padx=(6, 4), ipady=2)
    put(label(parent, suffix), side="left", padx=(0, 14))
    return box


def card(parent, title, subtitle):
    outer = put(
        tk.Frame(parent, bg=CARD_BG, highlightbackground=CARD_BORDER, highlightthickness=1),
        fill="x", padx=18, pady=8,
    )
    header = put(frame(outer), fill="x", padx=16, pady=(12, 8))
    put(frame(header, bg=ACCENT, width=3), side="left", fill="y", padx=(0, 10))
    text_col = put(frame(header), side="left", fill="x", expand=True)
    put(label(text_col, spaced(title.upper()), size=10, bold=True), anchor="w")
    put(label(text_col, subtitle, muted=True, size=8), anchor="w")
    return put(frame(outer), fill="both", expand=True, padx=16, pady=(0, 14))


def path_field(body, var, actions):
    row = put(frame(body), fill="x")
    put(entry(row, var), side="left", fill="x", expand=True, ipady=4)
    return [put(button(row, text, cmd, small=True), side="left", padx=(8, 0)) for text, cmd in actions]


class BatchGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"phantomsplit {APP_VERSION}")
        if ICON_PATH.exists():
            self.iconbitmap(default=str(ICON_PATH))
        self.geometry("820x720")
        self.minsize(680, 560)
        self.configure(bg=BG)
        enable_windows_dark_titlebar(self)

        cfg = load_config()
        self.vars = {
            key: (tk.BooleanVar if isinstance(value, bool) else tk.StringVar)(value=value)
            for key, value in cfg.items()
        }
        self.log_queue = queue.Queue()
        self.worker = None

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "Accent.Horizontal.TProgressbar", troughcolor=CARD_BG, background=ACCENT,
            bordercolor=CARD_BG, lightcolor=ACCENT, darkcolor=ACCENT, thickness=6,
        )

        self.build_widgets()
        self.set_input_type(cfg["input_type"])
        self.update_disk_label()
        self.set_status(False)
        self.vars["out"].trace_add("write", lambda *a: self.update_disk_label())
        self.after(100, self.poll_log_queue)

    def build_widgets(self):
        v = self.vars
        footer = put(
            frame(self, highlightbackground=CARD_BORDER, highlightthickness=1),
            side="bottom", fill="x", padx=18, pady=14,
        )
        put(button(footer, "save config", self.on_save, small=True), side="left", padx=14, pady=12)
        for text, url in LINKS:
            link = put(label(footer, text, muted=True, cursor="hand2"), side="left", padx=(0, 14))
            link.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))
            link.bind("<Enter>", lambda e: e.widget.configure(fg=ACCENT))
            link.bind("<Leave>", lambda e: e.widget.configure(fg=FG_MUTED))
        self.run_btn = put(button(footer, spaced("process"), self.on_run, accent=True), side="right", padx=14, pady=12)

        body = self.build_scroll_area()
        topbar = put(frame(body), fill="x", padx=18, pady=(16, 4))
        wordmark = put(frame(topbar), side="left")
        for text, kw in (("PHANTOM", {}), (" • ", {"fg": ACCENT}), ("SPLIT", {"muted": True})):
            put(label(wordmark, text, size=13, bold=True, **kw), side="left")
        dots = put(tk.Canvas(topbar, width=60, height=14, bg=BG, highlightthickness=0), side="right")
        for i, color in enumerate([ACCENT, DOT_OFF, DOT_OFF]):
            dots.create_oval(2 + i * 22, 2, 12 + i * 22, 12, fill=color, outline="")
        put(label(body, f"v{APP_VERSION}  •  {gpu_info()}", muted=True, size=8), anchor="w", padx=22, pady=(0, 6))

        mode = card(
            body, "process mode",
            "mixture files, or a folder of one song's stems to mix down",
        )
        row = put(frame(mode), anchor="w")
        self.mode_buttons = {
            name: put(button(row, name, lambda n=name: self.set_input_type(n)), side="left", padx=(0, 8))
            for name in ("mixture", "stems")
        }
        self.recursive_check = put(
            check(mode, "recursive (search subfolders)", v["recursive"]), anchor="w", pady=(10, 0),
        )
        inference = card(
            body, "inference", f"{MODEL_NAME}: more overlap is slower, a bigger batch needs more gpu memory",
        )
        row = put(frame(inference), anchor="w")
        for key in ("batch_size", "num_overlap"):
            field(row, key, v[key])
        put(
            check(inference, "tta (also split the song played backwards and average, 2x processing time)", v["tta"]),
            anchor="w", pady=(10, 0),
        )

        source = card(body, "input", "mixture file/folder, or in stems mode a folder of one song's stems")
        self.browse_file_btn = path_field(source, v["root"], [
            ("browse file", lambda: self.browse("root", folder=False)),
            ("browse folder", lambda: self.browse("root")),
        ])[0]

        output = card(body, "output", "where center.flac / wide.flac are written")
        path_field(output, v["out"], [
            ("browse", lambda: self.browse("out")),
            ("open folder", lambda: open_folder(v["out"].get())),
        ])
        self.disk_label = put(label(output, "", muted=True, size=8), anchor="w", pady=(10, 0))
        put(
            check(
                output,
                "flat mode (songname_center.flac / songname_wide.flac all in one folder, no per-song folders)",
                v["flat"],
            ),
            anchor="w", pady=(6, 0),
        )

        row = put(frame(card(body, "status", "current batch state")), fill="x")
        self.status_dot = put(tk.Canvas(row, width=10, height=10, bg=CARD_BG, highlightthickness=0), side="left")
        self.status_oval = self.status_dot.create_oval(1, 1, 9, 9, fill=DOT_OFF, outline="")
        self.status_label = put(
            label(row, "ready", muted=True, anchor="w"), side="left", padx=(6, 0), fill="x", expand=True,
        )
        self.progress = put(
            ttk.Progressbar(row, mode="determinate", style="Accent.Horizontal.TProgressbar"),
            side="left", fill="x", expand=True, padx=12,
        )
        self.progress_label = put(label(row, "", muted=True, width=16, anchor="e"), side="left")
        self.progress_mark = None

    def build_scroll_area(self):
        canvas = put(
            tk.Canvas(put(frame(self), side="top", fill="both", expand=True), bg=BG, highlightthickness=0),
            side="left", fill="both", expand=True,
        )
        scroll_frame = frame(canvas)
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        window_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(window_id, width=e.width))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        return scroll_frame

    def values(self):
        return {key: var.get() for key, var in self.vars.items()}

    def set_input_type(self, name):
        self.vars["input_type"].set(name)
        for mode, btn in self.mode_buttons.items():
            bg, fg, active = toggle_colors(mode == name, inactive_fg=FG_MUTED)
            btn.configure(bg=bg, fg=fg, activebackground=active)
        state = "disabled" if name == "stems" else "normal"
        self.browse_file_btn.configure(state=state)
        self.recursive_check.configure(state=state)

    def set_status(self, running):
        self.status_dot.itemconfig(self.status_oval, fill=ACCENT if running else DOT_OFF)
        self.status_label.configure(text="running" if running else "ready")

    def browse(self, key, folder=True):
        if folder:
            path = filedialog.askdirectory()
        else:
            patterns = " ".join(f"*{ext}" for ext in AUDIO_EXTS)
            path = filedialog.askopenfilename(filetypes=[("audio files", patterns), ("all files", "*.*")])
        if path:
            self.vars[key].set(path)

    def update_disk_label(self):
        out_str = self.vars["out"].get()
        free = disk_free_gb(out_str) if out_str and Path(out_str).exists() else None
        self.disk_label.configure(text=f"free space: {free:.1f} gb" if free is not None else "")

    def on_save(self):
        save_config(self.values())

    def inference_settings(self):
        batch_size, num_overlap = int(self.vars["batch_size"].get()), int(self.vars["num_overlap"].get())
        if batch_size < 1 or num_overlap < 1:
            raise ValueError
        return batch_size, num_overlap

    def update_status_message(self, msg):
        print(msg)
        self.status_label.configure(text=msg)

    def poll_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                if isinstance(msg, tuple):
                    done, total = msg
                    self.progress.configure(maximum=total, value=done)
                    now = time.monotonic()
                    if not self.progress_mark or done < self.progress_mark[2]:
                        self.progress_mark = (now, done, done)
                    started, first, _ = self.progress_mark
                    self.progress_mark = (started, first, done)
                    text = f"{100 * done // total}%"
                    if first < done < total:
                        left = round((now - started) * (total - done) / (done - first))
                        text += f"  {left // 60}:{left % 60:02d} left"
                    self.progress_label.configure(text=text)
                elif msg is None:
                    self.run_btn.configure(state="normal")
                    self.progress.configure(value=0)
                    self.progress_label.configure(text="")
                    self.progress_mark = None
                    self.status_dot.itemconfig(self.status_oval, fill=DOT_OFF)
                else:
                    self.update_status_message(msg)
        except queue.Empty:
            pass
        self.after(100, self.poll_log_queue)

    def validate(self):
        v = self.values()
        if not v["root"] or not Path(v["root"]).exists():
            return "input file or folder does not exist"
        if v["input_type"] == "stems" and not Path(v["root"]).is_dir():
            return "stems mode requires a folder of stem files"
        if not v["out"]:
            return "output folder is required"
        try:
            self.inference_settings()
        except ValueError:
            return "batch_size and num_overlap must be whole numbers, 1 or more"
        return None

    def on_run(self):
        if self.worker and self.worker.is_alive():
            return
        error = self.validate()
        if error:
            messagebox.showerror("phantomsplit", error)
            return

        self.on_save()
        v, settings = self.values(), self.inference_settings()
        self.run_btn.configure(state="disabled")
        self.set_status(True)
        emit = self.log_queue.put

        def on_progress(done, total):
            emit((done, total))

        def work():
            try:
                download_model_files(emit, on_progress)
                emit(f"loading {MODEL_NAME}")
                engines = load_engines(*settings, tta=v["tta"])
                engines[0]["on_progress"] = on_progress
                run_batch(
                    v["root"], v["out"], v["recursive"], engines, emit,
                    stems_mode=v["input_type"] == "stems", flat=v["flat"],
                )
            except Exception as e:
                print(traceback.format_exc())
                emit(f"[error] {e}")
            emit(None)

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()


if __name__ == "__main__":
    if os.name == "nt":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    BatchGUI().mainloop()
