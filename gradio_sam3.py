import subprocess
import time
import gc
from pathlib import Path
from typing import List, Tuple

import cv2
import gradio as gr
import numpy as np
import torch

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

    raise ValueError("비디오를 업로드하거나 YouTube URL을 입력하세요.")


def _extract_first_frame(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"첫 프레임을 읽을 수 없습니다: {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _format_points(points: List[Tuple[float, float]]) -> str:
    if not points:
        return "포인트가 없습니다."
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
            raise RuntimeError(f"비디오 파일이 존재하지 않습니다: {video_path}")

        frame = _extract_first_frame(video_path)
        h, w = frame.shape[:2]
        msg = f"✅ 비디오 준비 완료: {video_path} (w={w}, h={h})\n포인트 모드는 프레임을 클릭해 좌표를 추가하세요."
        return frame, video_path, [], (h, w), video_path, msg, _format_points([]), frame
    except Exception as e:
        return None, None, [], None, None, f"❌ 비디오 불러오기 실패: {e}", _format_points([]), None


def record_point(evt: gr.SelectData, points: List[Tuple[float, float]], frame_hw: Tuple[int, int] | None, frame_image: np.ndarray | None):
    """Append clicked point (x,y) in pixel space and draw a red dot."""
    if frame_hw is None or frame_image is None:
        return points, _format_points(points), "❌ 비디오를 먼저 불러오세요.", frame_image

    x, y = (evt.index or (None, None))
    if x is None or y is None:
        return points, _format_points(points), "❌ 좌표를 읽지 못했습니다. 다시 클릭하세요.", frame_image

    new_points = points + [(float(x), float(y))]
    vis = _draw_points_on_frame(frame_image, new_points)
    return new_points, _format_points(new_points), f"📍 포인트 추가: ({int(x)}, {int(y)})", vis


def clear_points(frame_image: np.ndarray | None):
    return [], _format_points([]), "🧹 포인트를 모두 지웠습니다.", frame_image


def run_sam(video_path: str | None, prompt_type: str, text_prompt: str, points: List[Tuple[float, float]]):
    if not video_path:
        return None, "❌ 비디오를 먼저 불러오세요."

    ts = int(time.time())
    save_path = OUTPUT_DIR / f"sam3_mask_{prompt_type}_{ts}.mp4"

    try:
        if prompt_type == "text":
            if not text_prompt.strip():
                raise ValueError("텍스트 프롬프트를 입력하세요.")
            from get_mask_text import get_mask_from_text
            get_mask_from_text(video_path, prompt=text_prompt.strip(), save_path=str(save_path))
            msg = f"✅ 텍스트 프롬프트 완료. 결과 저장: {save_path}"
        else:
            if not points:
                raise ValueError("포인트를 한 개 이상 추가하세요.")
            from get_mask_point import get_mask_from_point
            get_mask_from_point(video_path, point_prompt=points, save_path=str(save_path))
            msg = f"✅ 포인트 프롬프트 완료. 결과 저장: {save_path}"
    except Exception as e:
        raise e
        return None, f"❌ 마스크 생성 실패: {e}"

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
        status = "✅ SAM 메모리 정리 완료" if (cleared_text or cleared_point) else "⚠️ 정리할 메모리가 없습니다"
    except Exception as e:
        return f"❌ SAM 메모리 정리 실패: {e}"
    return status


def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("### SAM3 Video Masking\n비디오를 업로드하거나 YouTube URL을 입력한 뒤 프롬프트 타입을 선택하세요. 포인트 모드에서는 첫 프레임을 클릭해 좌표를 추가합니다.")

        video_path_state = gr.State(value=None)
        points_state = gr.State(value=[])
        frame_hw_state = gr.State(value=None)
        frame_image_state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                video_file = gr.Video(
                    label="비디오 업로드 (MP4)",
                    sources=["upload"],
                )
                youtube_url = gr.Textbox(label="YouTube URL (선택)", placeholder="https://www.youtube.com/...")
                load_btn = gr.Button("비디오 불러오기 / 다운로드 (비디오 업로드 후 클릭하세요)", variant="secondary")
                load_status = gr.Markdown("비디오를 불러와 첫 프레임을 확인하세요.")

                prompt_type = gr.Radio(
                    ["text", "point"],
                    value="text",
                    label="프롬프트 타입",
                )
                text_prompt = gr.Textbox(label="텍스트 프롬프트", placeholder="예: person", visible=True)

                point_frame = gr.Image(
                    label="프레임에서 포인트 클릭 (첫 프레임)",
                    type="numpy",
                    interactive=True,
                    visible=False,
                )
                points_display = gr.Markdown(_format_points([]), visible=False)
                clear_points_btn = gr.Button("포인트 지우기", variant="secondary", visible=False)
                clear_sam_btn = gr.Button("SAM 메모리 비우기", variant="secondary")

                submit_btn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                result_video = gr.Video(label="마스크 결과", interactive=False)
                status = gr.Markdown(label="상태")

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
