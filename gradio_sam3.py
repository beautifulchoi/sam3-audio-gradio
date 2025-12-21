import subprocess
import time
import gc
from pathlib import Path
from typing import List, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
import os
from utils.get_youtube import download_youtube


def _path_from_uploaded(uploaded_file):
    """Robustly extract a local file path from Gradio upload variants."""
    if uploaded_file is None:
        return None
    # Gradio can pass a list/tuple of paths or file dicts
    if isinstance(uploaded_file, (list, tuple)) and uploaded_file:
        return _path_from_uploaded(uploaded_file[0])
    # Direct string path
    if isinstance(uploaded_file, (str, Path)):
        return str(uploaded_file)
    # Dict with name/path keys
    if isinstance(uploaded_file, dict):
        for key in ("path", "name", "tempfile", "data"):
            if key in uploaded_file and uploaded_file[key]:
                val = uploaded_file[key]
                if isinstance(val, (str, Path)):
                    return str(val)
        return None
    # File-like object
    if hasattr(uploaded_file, "name"):
        return str(uploaded_file.name)
    return None

DOWNLOAD_DIR = Path("downloads")
OUTPUT_DIR = Path("outputs")
DOWNLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def _resolve_video_path(uploaded_file, youtube_url: str) -> str:
    """Pick uploaded file first; otherwise download from YouTube."""
    path = _path_from_uploaded(uploaded_file)
    if path:
        return str(path)

    if youtube_url and youtube_url.strip():
        downloads = download_youtube(youtube_url.strip(), str(DOWNLOAD_DIR))
        return downloads["video_only"]

    raise ValueError("Please upload a video or enter a YouTube URL.")


def _extract_first_frame(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read the first frame: {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _format_points(points: List[Tuple[float, float]]) -> str:
    if not points:
        return "No points."
    return "\n".join([f"{idx + 1}. ({int(x)}, {int(y)})" for idx, (x, y) in enumerate(points)])


def _draw_points_on_frame(frame: np.ndarray, points: List[Tuple[float, float]]):
    if frame is None:
        return None
    vis = frame.copy()
    for x, y in points:
        cv2.circle(vis, (int(x), int(y)), radius=4, color=(255, 0, 0), thickness=-1)  # red dot
    return vis


def load_video(video_file, youtube_url):
    """Load/upload/download video, grab first frame for point selection."""
    try:
        video_path = _resolve_video_path(video_file, youtube_url)
        # Normalize/ensure playable (H.264/yuv420p, limited size) for browser playback

        video_path = str(Path(video_path).resolve())
        if not Path(video_path).exists():
            raise RuntimeError(f"No video file exist: {video_path}")

        frame = _extract_first_frame(video_path)
        h, w = frame.shape[:2]
        msg = f"✅ Video ready: {video_path} (w={w}, h={h})\nIn point mode, click the frame to add coordinates."
        return frame, video_path, [], (h, w), video_path, msg, _format_points([]), frame
    except Exception as e:
        return None, None, [], None, None, f"❌ fail to load the video: {e}", _format_points([]), None


def record_point(evt: gr.SelectData, points: List[Tuple[float, float]], frame_hw: Tuple[int, int] | None, frame_image: np.ndarray | None):
    """Append clicked point (x,y) in pixel space and draw a red dot."""
    if frame_hw is None or frame_image is None:
        return points, _format_points(points), "❌ Please load a video first.", frame_image

    x, y = (evt.index or (None, None))
    if x is None or y is None:
        return points, _format_points(points), "❌ Could not read coordinates. Please click again.", frame_image

    new_points = points + [(float(x), float(y))]
    vis = _draw_points_on_frame(frame_image, new_points)
    return new_points, _format_points(new_points), f"📍 Point added: ({int(x)}, {int(y)})", vis


def clear_points(frame_image: np.ndarray | None):
    return [], _format_points([]), "🧹 All points cleared.", frame_image


def run_sam(video_path: str | None, prompt_type: str, text_prompt: str, points: List[Tuple[float, float]]):
    if not video_path:
        return None, "❌ load video first."

    video_base = os.path.splitext(os.path.basename(video_path))[0]
    save_path = OUTPUT_DIR / f"{video_base}_mask.mp4"

    try:
        if prompt_type == "text":
            if not text_prompt.strip():
                raise ValueError("Please enter a text prompt.")
            from get_mask_text import get_mask_from_text
            get_mask_from_text(video_path, prompt=text_prompt.strip(), save_path=str(save_path))
            msg = f"✅ Text prompt complete. Result saved: {save_path}"
        else:
            if not points:
                raise ValueError("Please add at least one point.")
            from get_mask_point import get_mask_from_point
            get_mask_from_point(video_path, point_prompt=points, save_path=str(save_path))
            msg = f"✅ Point prompt complete. Result saved: {save_path}"
    except Exception as e:
        raise e
        return None, f"❌ Mask creation failed: {e}"

    return str(save_path), msg


def clear_sam_memory():
    try:
        from get_mask_text import clear_sam as clear_text
        from get_mask_point import clear_sam_point
        cleared_text = bool(clear_text())
        cleared_point = bool(clear_sam_point())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        status = "✅ SAM memory cleared." if (cleared_text or cleared_point) else "⚠️ No memory to clear"
    except Exception as e:
        return f"❌ Failed to clear SAM memory: {e}"
    return status


def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("### SAM3 Video Masking\nUpload a video or enter a YouTube URL, then select a prompt type. In point mode, click the first frame to add coordinates.")
        status_text = gr.Textbox(label="Status", interactive=False, value="Ready.")

        video_path_state = gr.State(value=None)
        points_state = gr.State(value=[])
        frame_hw_state = gr.State(value=None)
        frame_image_state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                video_file = gr.Video(
                    label="Video upload (MP4)",
                    sources=["upload"],
                )

                youtube_url = gr.Textbox(label="YouTube URL (optional)", placeholder="https://www.youtube.com/...")
                load_btn = gr.Button("Load / Download video (After upload, click this button)", variant="secondary")
                load_status = gr.Markdown("Load a video and check the first frame.")

                prompt_type = gr.Radio(
                    ["text", "point"],
                    value="text",
                    label="Prompt Type",
                )
                text_prompt = gr.Textbox(label="Text Prompt", placeholder="e.g., person", visible=True)

                point_frame = gr.Image(
                    label="Click points on the first frame",
                    type="numpy",
                    interactive=True,
                    visible=False,
                )
                points_display = gr.Markdown(_format_points([]), visible=False)
                clear_points_btn = gr.Button("Clear Points", variant="secondary", visible=False)
                clear_sam_btn = gr.Button("Clear SAM Memory", variant="secondary")

                submit_btn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                result_video = gr.Video(label="Mask Result", interactive=False)
                # Show status bar in the output column as well
                status_text
                status = gr.Markdown(label="status")

        # Toggle UI for text/point prompts
        prompt_type.change(
            lambda t: (
                gr.update(visible=t == "text"),
                gr.update(visible=t == "point"),
                gr.update(visible=t == "point"),
                gr.update(visible=t == "point"),
            ),
            inputs=prompt_type,
            outputs=[text_prompt, point_frame, points_display, clear_points_btn],
        )

        # Load video (upload or YouTube) and extract first frame
        load_btn.click(
            load_video,
            inputs=[video_file, youtube_url],
            outputs=[
                point_frame,
                video_path_state,
                points_state,
                frame_hw_state,
                result_video,
                load_status,
                points_display,
                frame_image_state,
            ],
        )

        # Capture point clicks on the frame
        point_frame.select(
            record_point,
            inputs=[points_state, frame_hw_state, frame_image_state],
            outputs=[points_state, points_display, status, point_frame],
        )

        clear_points_btn.click(
            clear_points,
            inputs=[frame_image_state],
            outputs=[points_state, points_display, status, point_frame],
        )

        clear_sam_btn.click(
            clear_sam_memory,
            inputs=[],
            outputs=[status],
        )

        submit_btn.click(
            run_sam,
            inputs=[video_path_state, prompt_type, text_prompt, points_state],
            outputs=[result_video, status],
        )
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
