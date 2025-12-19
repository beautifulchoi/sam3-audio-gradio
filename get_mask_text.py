from utils.load_sam import get_sam
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)
import cv2
import numpy as np
import torch
import gc
from utils.func import *

bpe_path = "/home/prj/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
video_predictor = build_sam3_video_predictor(bpe_path=bpe_path, gpus_to_use=[1])

def init_mask(video_path:str, prompt): # a JPEG folder or an MP4 video file
    # Start a session
    response = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response['session_id']
    request = dict(
            type="add_prompt",
            session_id=response["session_id"],
            frame_index=0, # Arbitrary frame index
            text=prompt,
            )

    response = video_predictor.handle_request(request=request)
    output = response["outputs"]

    return output, session_id

def propagate_in_video(session_id):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in video_predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame

def get_mask_from_text(video_path, prompt, save_path):
    _, session_id = init_mask(video_path, prompt=prompt)
    output = propagate_in_video(session_id)
    output = prepare_masks_for_visualization(output)
    get_sam_video(output, video_path, save_path=save_path)

def clear_sam():
    global video_predictor
    try:
        del video_predictor
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True

if __name__ =="__main__":
    video_path = "/home/prj/data/sample_videos/snowboard_832.mp4"
    prompt = 'person'
    save_path='./sample_text.mp4'
    get_mask_from_text(video_path, prompt, save_path)