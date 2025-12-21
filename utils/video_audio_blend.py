import cv2
import numpy as np
import torch
import torchaudio
import os

def blend_video_with_mask_and_audio(video_path, mask_path, audio_path, output_path, alpha=0.5):
    """
    Blend video with mask and add separated audio.
    Args:
        video_path: Path to original video
        mask_path: Path to mask video (same frame count/size)
        audio_path: Path to separated audio (wav)
        output_path: Path to save the output video
        alpha: Blend ratio for mask overlay
    """
    # Open video and mask
    cap_vid = cv2.VideoCapture(video_path)
    cap_mask = cv2.VideoCapture(mask_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap_vid.get(cv2.CAP_PROP_FPS)
    width = int(cap_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret_vid, frame = cap_vid.read()
        ret_mask, mask_frame = cap_mask.read()
        if not ret_vid or not ret_mask:
            break
        # Convert mask to grayscale and normalize
        mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
        mask_norm = (mask_gray / 255.0)[:, :, None]
        # Blend: mask highlights in red
        overlay = frame.copy()
        overlay[:, :, 2] = np.clip(overlay[:, :, 2] + 128 * mask_norm[:, :, 0], 0, 255)
        blended = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        out.write(blended.astype(np.uint8))
    cap_vid.release()
    cap_mask.release()
    out.release()

    # Add audio using ffmpeg
    temp = output_path + ".tmp.mp4"
    os.rename(output_path, temp)
    cmd = f"ffmpeg -y -i {temp} -i {audio_path} -c:v copy -c:a aac -shortest {output_path}"
    os.system(cmd)
    os.remove(temp)
