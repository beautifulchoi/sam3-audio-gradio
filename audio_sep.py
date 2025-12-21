from sam_audio import SAMAudio, SAMAudioProcessor
import torchaudio
import cv2
import torch
import numpy as np
import os

def load_input(video_file, mask_file):
    cap_video = cv2.VideoCapture(video_file)
    height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap_video.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    while cap_video.isOpened():
        ret, frame = cap_video.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame).permute(2, 0, 1))
    cap_video.release()

    frames = torch.stack(frames)
    print(f"Video loaded: {frames.shape} (T, C, H, W), FPS: {fps}")

    # Load mask video
    cap_mask = cv2.VideoCapture(mask_file)
    masks = []
    while cap_mask.isOpened():
        ret, mask_frame = cap_mask.read()
        if not ret:
            break
        # Convert to grayscale and threshold to binary
        mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
        binary_mask = (mask_gray > 127).astype(bool)
        masks.append(torch.from_numpy(binary_mask))
    cap_mask.release()

    mask = torch.stack(masks).unsqueeze(1)  # Shape: (T, 1, H, W)
    print(f"Mask loaded: {mask.shape} (T, 1, H, W)")

    # Verify dimensions match
    assert frames.shape[0] == mask.shape[0], f"Frame count mismatch: video={frames.shape[0]}, mask={mask.shape[0]}"
    assert frames.shape[2:] == mask.shape[2:], f"Dimension mismatch: video={frames.shape[2:]}, mask={mask.shape[2:]}"

    return frames, mask

def sep_audio(model, processor, video_file, frames, mask, output_dir="output_sep"):
    inputs = processor(
        audios=[video_file],
        descriptions=[""],
        masked_videos=processor.mask_videos([frames], [mask]),
    ).to('cuda')
    with torch.inference_mode():
        result = model.separate(inputs)

    sep = result.target[0].cpu().float()
    res = result.residual[0].cpu().float()
    # Ensure correct shape: [channels, samples]
    if sep.ndim == 1:
        sep = sep.unsqueeze(0)
    if res.ndim == 1:
        res = res.unsqueeze(0)
    # Save separated audio (target)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(video_file))[0]
        target_audio_path = os.path.join(output_dir, f"{base}_target.wav")
        torchaudio.save(
            target_audio_path,
            sep,
            sample_rate=processor.audio_sampling_rate
        )
        print(f"Target audio saved to: {target_audio_path}")

        # Save residual audio (background)
        residual_audio_path = os.path.join(output_dir, f"{base}_residual.wav")
        torchaudio.save(
            residual_audio_path,
            res,
            sample_rate=processor.audio_sampling_rate
        )
        print(f"Residual audio saved to: {residual_audio_path}")
    return sep, res

if __name__ == "__main__":
    model = SAMAudio.from_pretrained("facebook/sam-audio-large")
    processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-large")
    model = model.eval().cuda()
    video_file = "/home/prj/play_sam3_audio/downloads/cUzeq_-hQ-o/cUzeq_-hQ-o_va_s720_f15.mp4"
    mask_file = "/home/prj/play_sam3_audio/outputs/sam3_mask_point_1766141782_mask.mp4"
    f, m = load_input(video_file, mask_file)
    sep_audio(model, processor, video_file, f, m, output_dir="output_sep")