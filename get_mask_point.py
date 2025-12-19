from utils.load_sam import get_sam
from sam3.model_builder import build_sam3_video_model
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)
import cv2
import numpy as np
import torch
import os
import gc
from utils.func import *

sam3_model = build_sam3_video_model(bpe_path="/home/prj/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
predictor = sam3_model.tracker
predictor.backbone = sam3_model.detector.backbone

def init_mask(video_path:str, prompt:list[tuple]): # a JPEG folder or an MP4 video file
    # Start a session
    inference_state = predictor.init_state(video_path=video_path)
    predictor.clear_all_points_in_video(inference_state)
    points = np.asarray(prompt, dtype=np.float32)
    labels = np.array([1]*len(prompt), np.int32)
    h,w = get_first_frame_hw(video_path)
    rel_points = [[x / w, y / h] for x, y in points]
    points_tensor = torch.tensor(rel_points, dtype=torch.float32)
    points_labels_tensor = torch.tensor(labels, dtype=torch.int32)

    obj_ids = 0

    _, out_obj_ids, low_res_masks, video_res_masks = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=obj_ids,
        points=points_tensor,
        labels=points_labels_tensor,
        clear_old_points=False,
    )

    return inference_state, out_obj_ids

def propagate_in_video(inference_state, out_obj_ids):
    # we will just propagate from frame 0 to the end of the video
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(inference_state, start_frame_idx=0, max_frame_num_to_track=300, reverse=False, propagate_preflight=True):
        video_segments[frame_idx] = {
            out_obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    return video_segments


def get_mask_from_point(video_path, point_prompt, save_path):
    inference_state, out_obj_ids = init_mask(video_path, prompt=point_prompt)
    output = propagate_in_video(inference_state, out_obj_ids)
    get_sam_video(output, video_path, save_path=save_path)


def clear_sam_point():
    global predictor, sam3_model
    try:
        del predictor
    except Exception:
        pass
    try:
        del sam3_model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True

if __name__ =="__main__":
    video_path = "/home/prj/data/sample_videos/snowboard_832.mp4"
    prompt = [[200,400]]
    inference_state, out_obj_ids = init_mask(video_path, prompt=[[200,400]])
    save_path='./sample_point.mp4'
    get_mask_from_point(video_path, prompt, save_path)