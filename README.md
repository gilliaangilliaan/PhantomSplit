# PhantomSplit

PhantomSplit splits stereo audio into a mono center track and a wide (side) track, using a BS-Roformer center/wide model. It runs as a small desktop app on Windows.

## Features

- **Mixture mode:** split a single file or a whole folder of songs into a center and a wide file.
- **Stems mode:** split every stem in a folder of one song, then mix the results down into one center and one wide file.
- **Flat mode:** write `songname_center` / `songname_wide` into one folder, with no per-song subfolders.
- **TTA:** also split the song played backwards and average the two results (about twice the processing time).
- **Input formats:** WAV, FLAC, MP3, AIFF, OGG, M4A / MP4 / AAC, Opus and WMA. Files that cannot be read are listed in a message at the end of the batch.
- **Output format:** WAV (16 / 24 / 32-bit float), FLAC (16 / 24-bit) or MP3 (128 to 320 kbps).
- **Batch size and overlap:** adjustable in the app; the defaults come from the model's config.
- Drag and drop a file or folder onto the window to set the input, or onto the output bar to set the output folder.
- Progress bar for the whole batch with the song position ("song 3 of 40"), percentage and time remaining, a stop button, and resume support (finished songs are skipped).

## Install

Download [PhantomSplit-1.0.1-Setup.exe](https://github.com/gilliaangilliaan/PhantomSplit/releases/download/1.0.1/PhantomSplit-1.0.1-Setup.exe) from the [1.0.1 release](https://github.com/gilliaangilliaan/PhantomSplit/releases/tag/1.0.1) and run it.

The installer:

1. Copies the app to `%LOCALAPPDATA%\PhantomSplit` (no admin rights needed).
2. Downloads Python 3.10 and creates a private environment for the app.
3. Installs the PyTorch build that matches your graphics card:
   - NVIDIA RTX 50 series: PyTorch 2.7.1 (CUDA 12.8)
   - other NVIDIA cards: PyTorch 2.5.1 (CUDA 12.1)
   - no NVIDIA card: PyTorch 2.5.1 CPU build (much slower)

The first time you press **process**, the app downloads the model (about 980 MB) from Hugging Face.

Requirements: 64-bit Windows 10 or 11, an internet connection during install, and about 7 GB of free disk space. An NVIDIA card is strongly recommended.

## Run from source

```
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r installer\requirements.txt
python phantomsplitnmdx.py
```

Use Python 3.10. Swap `cu121` for `cu128` (with `torch==2.7.1`) on an RTX 50 series card, or for `cpu` if you have no NVIDIA card.

## Build the installer

Install [Inno Setup](https://jrsoftware.org/isinfo.php), then:

```
"C:\Program Files\Inno Setup 7\ISCC.exe" installer\PhantomSplit.iss
```

The setup file is written to `installer\Output`.

## Model

The center/wide model is downloaded from [gilliaan/CenterWide-Models](https://huggingface.co/gilliaan/CenterWide-Models/tree/main/BandSplitSpatialFormer) on Hugging Face.

## Credits

- The BS-Roformer model code in `bs_roformer.py` and `modules/bs_roformer` is adapted from [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training).

## Links

- GitHub: https://github.com/gilliaangilliaan
- Ko-fi: https://ko-fi.com/gilliaan
- Hugging Face: https://huggingface.co/gilliaan
