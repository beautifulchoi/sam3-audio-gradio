import cv2
import os
import numpy as np
import torch

def get_first_frame_hw(video_path):
    """Return (height, width) of the first frame in the video."""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read first frame from {video_path}")
    h, w = frame.shape[:2]
    return h, w

def abs_to_rel_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="point"):
    """Convert absolute coordinates to relative coordinates (0-1 range)

    Args:
        coords: List of coordinates
        coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
    """
    if coord_type == "point":
        return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [
            [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
            for x, y, w, h in coords
        ]
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")

def _make_color_palette(n_colors=12):
    """Generate a simple rainbow-ish palette in RGB."""
    hsvs = [(i / n_colors, 1.0, 1.0) for i in range(n_colors)]
    colors = []
    for h, s, v in hsvs:
        rgb = np.array(cv2.cvtColor(np.uint8([[[h * 179, s * 255, v * 255]]]), cv2.COLOR_HSV2BGR), dtype=np.uint8)[0, 0]
        colors.append((int(rgb[2]), int(rgb[1]), int(rgb[0])))  # convert BGR->RGB order
    return colors

def get_sam_video(outputs_per_frame, video_path, alpha=0.5, save_path=None, save_mask=True, fps=None):
    """Blend SAM masks onto the original video frames.

    outputs_per_frame: {frame_idx: {object_id: mask}} where mask is a 2D bool/0-1 array.
    alpha: blend strength for the mask color.
    save_path: optional mp4 path to write blended video.
    fps: override output FPS; if None, uses source FPS.
    """
    frames = process_video(video_path)
    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")

    palette = _make_color_palette(24)
    blended_frames = []

    for idx, frame in enumerate(frames):
        if idx not in outputs_per_frame:
            blended_frames.append(frame)
            continue

        overlay = frame.copy()
        for obj_id, mask in outputs_per_frame[idx].items():
            if mask is None:
                continue
            if mask.ndim ==3 and mask.shape[0]==1:
                mask = mask.squeeze()
            if mask.shape != overlay.shape[:2]:
                # resize mask if needed
                mask = cv2.resize(mask.astype(float), 
                                  (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

            obj_idx = int(obj_id)

            color = palette[obj_idx % len(palette)]
            overlay[mask] = color

        blended = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        blended_frames.append(blended)

    if save_path:
        cap = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        fps_out = fps or src_fps
        h, w, _ = blended_frames[0].shape
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (w, h))

        mask_writer = None
        if save_mask:
            mask_path = os.path.splitext(save_path)[0] + "_mask.mp4"
            mask_writer = cv2.VideoWriter(mask_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (w, h), isColor=False)

        for idx, f in enumerate(blended_frames):
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            if mask_writer is not None:
                if idx not in outputs_per_frame:
                    mask_writer.write(np.zeros((h, w), dtype=np.uint8))
                else:
                    combined_mask = np.zeros((h, w), dtype=np.uint8)
                    for obj_id, mask in outputs_per_frame[idx].items():
                        if mask is None:
                            continue
                        mask_arr = np.asarray(mask).squeeze()
                        mask_bool = mask_arr > 0.5 if mask_arr.dtype != bool else mask_arr
                        if mask_bool.shape[:2] != (h, w):
                            mask_bool = cv2.resize(mask_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                        combined_mask[mask_bool] = 255
                    mask_writer.write(combined_mask)

        writer.release()
        if mask_writer is not None:
            mask_writer.release()

    return blended_frames

def process_video(video_path):
    if isinstance(video_path, str) and video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        video_frames_for_vis = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

    return video_frames_for_vis
