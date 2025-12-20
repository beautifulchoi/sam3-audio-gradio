# SAM3 Audio Gradio Demo

Minimal Gradio UI to run SAM3 video masking with either text prompts or point clicks on the first frame. You can upload an MP4 or supply a YouTube URL (auto-downloads, re-encodes to H.264, and downscales to ease GPU load). Outputs are written to `outputs/` as MP4.

> SAM Audio integration is still in progress; current UI covers SAM3 video masking only.

## Prerequisites
- Python 3.10+ and `ffmpeg` available on PATH.
- CUDA GPU recommended; CPU is not tested.
- Submodules checked out: `sam-audio` and `sam3`.

## Setup
```bash
# Clone the repo
git clone https://github.com/beautifulchoi/sam3-audio-gradio.git
cd sam3-audio-gradio

# Option A: pull submodules (preferred)
git submodule update --init --recursive
# Option B: if submodules are missing, clone/install via script
./download_sam.sh

# (Optional) create and activate a virtual env
python -m venv .venv
source .venv/bin/activate

# Install main Python deps (requirements are WIP; follow sam-audio for now)
pip install -r sam-audio/requirements.txt || pip install sam-audio
# sam3 requirements are still being aligned; if needed, mirror packages from sam-audio
pip install gradio opencv-python-headless yt-dlp torch torchvision torchaudio
```

> Note: some SAM/SAM3 models may require authenticated downloads. Follow each submodule's README if access is needed.

## Run the app
```bash
python gradio_sam3.py
```
Gradio will print a local URL; open it in your browser.

## How to use (UI)
1. Upload an MP4 **or** paste a YouTube URL.
2. Click **비디오 불러오기 / 다운로드** to fetch and normalize the video; the first frame will appear.
3. Choose **text** or **point** prompt type:
   - **text**: enter a text prompt (e.g., `person`).
   - **point**: click one or more locations on the first frame; added points list appears.
4. Click **Submit**. A masked video renders on the right and saves under `outputs/`.
5. Use **SAM 메모리 비우기** if the model gets OOM or you want to clear cached weights.

## Notes
- Downloads and intermediates go to `downloads/`; results go to `outputs/`.
- YouTube fetch forces H.264/yuv420p, downscales to 720p/15fps, and muxes audio back to keep playback smooth.
- For point prompts, masks are computed over the whole video using the selected coordinates from the first frame.
- If you see codec or GPU errors, re-run with smaller videos or adjust downscale FPS/height in `utils/get_youtube.py`.

## TODO
- [x] Connect SAM3 with Gradio
- [ ] Support multiple point clicks
- [ ] Extend with SAM Audio
- [ ] Integrate SAM3–SAM Audio in Gradio
- [ ] Automatic dataset extraction tool
