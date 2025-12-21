
# SAM3 Audio Gradio Demo

Minimal Gradio UI to run SAM3 video masking and **SAM Audio separation** with either text prompts or point clicks on the first frame. You can upload an MP4 or supply a YouTube URL (auto-downloads, re-encodes to H.264, and downscales to ease GPU load). Outputs are written to `outputs/` as MP4 and WAV.

**New:**
- SAM Audio separation is now fully integrated! You can extract target and residual audio from masked videos.
- You can also blend the mask with the video and add the separated audio, producing a new video for easy review.

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

## Check Whether download correctly
To verify that the SAM models and dependencies are properly downloaded and accessible, run the following test:
```bash
python utils/load_sam.py
```
If you encounter errors, ensure that:
- Submodules are properly initialized (`git submodule update --init --recursive`)
- Model checkpoint files are downloaded (some may require authentication)
- Required Python packages are installed



## Run the app to extract the mask
```bash
python gradio_sam3.py
```

## Run the app to separate the audio
```bash
python gradio_audio.py
```
Gradio will print a local URL; open it in your browser.

## How to use (UI)
### For SAM3 Masking (gradio_sam3.py)
1. Upload an MP4 **or** paste a YouTube URL.
2. Click **load / download video** to fetch and normalize the video; the first frame will appear.
3. Choose **text** or **point** prompt type:
   - **text**: enter a text prompt (e.g., `person`).
   - **point**: click one or more locations on the first frame; added points list appears.
4. Click **Submit**. A masked video renders on the right and saves under `outputs/`.
5. Use **SAM memory clear** if the model gets OOM or you want to clear cached weights.

### For SAM Audio Separation (gradio_audio.py)
1. Upload a video file and its corresponding mask video (from SAM3 masking step).
2. Click **Separate Audio** (button is enabled after model loads).
3. Download the separated target and residual audio files (WAV).
4. (Optional) Click **Create Blended Video** to generate a video with the mask blended and separated audio overlaid.
5. Use **Clear Memory** if you encounter GPU memory issues.

## Notes
- During the SAM3 Gradio process, all downloads and intermediate files are saved in the `downloads/` folder; final results are saved in `outputs/`.
- YouTube downloads are automatically re-encoded to H.264/yuv420p, downscaled to 720p/15fps, and audio is muxed back in for smooth playback.
- For point prompts, the mask is computed for the entire video using the coordinates you select on the first frame.
- If you encounter codec or GPU errors, try using a smaller video or adjust the downscale FPS/height settings in `utils/get_youtube.py`.
- After extracting a mask, locate the output mask file (`*_mask.mp4`) and the original video (with both audio and video), then use these as inputs for the audio separation app (`gradio_audio.py`).