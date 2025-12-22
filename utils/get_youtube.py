import os
import subprocess
from pathlib import Path
import yt_dlp
from urllib.parse import urlparse

def run_ffmpeg(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8"))


def _downscale_video(video_path: str, max_height: int = 720, target_fps: int = 15) -> str:
    """Downscale video to reduce GPU load (H.264/yuv420p)."""
    src = Path(video_path)
    dst = src.with_name(src.stem + f"_s{max_height}_f{target_fps}.mp4")
    # If a previous run left a zero-byte or corrupted file, regenerate it
    if dst.exists() and dst.stat().st_size > 0:
        return str(dst)
    if dst.exists():
        dst.unlink()

    # Use scale width -2 to keep aspect ratio and guarantee even dimensions for H.264
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vf",
        f"scale=-2:'min({max_height},ih)'",
        "-r",
        str(target_fps),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(dst),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8"))
    return str(dst)

def download_youtube(url, download_root, clip_seconds=None, ar=None, reencode_h264=True, downscale=True, max_height=720, target_fps=15):
    parsed = urlparse(url)
    base_name = os.path.basename(parsed.path) or "download"

    out_dir = os.path.join(download_root, base_name)
    os.makedirs(out_dir, exist_ok=True)

    # Targets
    out_base = os.path.join(out_dir, f"{base_name}_va")
    va_mp4 = f"{out_base}.mp4"            # merged video+audio (target container)
    audio_wav = os.path.join(out_dir, f"{base_name}_audio.wav")
    video_mp4 = os.path.join(out_dir, f"{base_name}_video.mp4")

    # 1) Download merged video+audio once
    ydl_opts = {
        # Prefer H.264 (avc1) to avoid AV1 decoder issues; fall back to best <=1080p
        "format": "bv*[vcodec^=avc1][height<=1080]+ba/b[height<=1080]",
        "outtmpl": f"{out_base}.%(ext)s",  # let yt_dlp set the downloaded ext, then merge to mp4
        "merge_output_format": "mp4",
    }
    print(f"[download] video+audio -> {va_mp4}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # 2) Extract audio (wav) from downloaded file
    cmd_audio = ["ffmpeg", "-y", "-i", va_mp4, "-vn", "-acodec", "pcm_s16le", "-ar", "44100"]
    if clip_seconds:
        cmd_audio += ["-t", str(clip_seconds)]
    if ar is not None:
        cmd_audio += ["-ar", str(ar)]  # optional resample

    if clip_seconds:
        cmd_audio += ["-t", str(clip_seconds)]
    cmd_audio += [audio_wav]
    print(f"[ffmpeg] extract audio -> {audio_wav}")
    run_ffmpeg(cmd_audio)

    # 3) Extract video-only (strip audio)
    cmd_video = ["ffmpeg", "-y", "-i", va_mp4, "-an"]
    if reencode_h264:
        # Force H.264 to avoid AV1 decode issues.
        cmd_video += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    else:
        cmd_video += ["-c:v", "copy"]
    if clip_seconds:
        cmd_video += ["-t", str(clip_seconds)]
    cmd_video += [video_mp4]
    print(f"[ffmpeg] extract video-only -> {video_mp4}")
    run_ffmpeg(cmd_video)

    # 4) Optional downscale to reduce GPU load
    video_mp4_final = video_mp4
    if downscale:
        video_mp4_final = _downscale_video(video_mp4, max_height=max_height, target_fps=target_fps)

    # 5) Merge downscaled (or original) video with audio to create va file
    va_mp4_final = f"{out_base}_s{max_height}_f{target_fps}.mp4" if downscale else va_mp4
    cmd_mux = [
        "ffmpeg",
        "-y",
        "-i",
        video_mp4_final,
        "-i",
        audio_wav,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        va_mp4_final,
    ]
    print(f"[ffmpeg] mux video+audio -> {va_mp4_final}")
    run_ffmpeg(cmd_mux)

    print("\n✔ Done")
    print(f"  video+audio (final): {va_mp4_final}")
    print(f"  audio-only        : {audio_wav}")
    print(f"  video-only        : {video_mp4_final}")
    return {
        "video_audio": va_mp4_final,
        "audio": audio_wav,
        "video_only": video_mp4_final,
        "video_audio_orig": va_mp4,
    }


if __name__ == "__main__":
    url = "https://www.youtube.com/shorts/cUzeq_-hQ-o"
    download_root = "/home/prj/audio_samples" # Save to current directory for easy testing
    
    
    download_youtube(url, download_root)
