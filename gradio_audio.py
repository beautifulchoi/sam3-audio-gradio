import gradio as gr
import torch
import os
from audio_sep import load_input, sep_audio
from sam_audio import SAMAudio, SAMAudioProcessor

# Global model and processor
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_choices = []
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(idx)
        gpu_choices.append((f"cuda:{idx}", f"GPU {idx}: {name}"))
gpu_choices.append(("cpu", "CPU (very slow)"))


def _clear_all_cuda():
    """Clear caches on all CUDA devices."""
    if not torch.cuda.is_available():
        return
    for idx in range(torch.cuda.device_count()):
        torch.cuda.set_device(idx)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_model_on_device(target_device: str):
    """Load SAM Audio on a specific device with OOM handling."""
    global model, processor, device
    # Drop previous model before retrying
    model = None
    processor = None
    if torch.cuda.is_available():
        _clear_all_cuda()
    try:
        print(f"Loading SAM Audio model on {target_device}...")
        if target_device.startswith("cuda"):
            torch.cuda.set_device(int(target_device.split(":")[1]))
        model = SAMAudio.from_pretrained("facebook/sam-audio-large")
        processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-large")
        model = model.eval().to(target_device)
        device = target_device
        msg = f"✅ Model loaded on {target_device}"
        print(msg)
        return msg, gr.update(interactive=True)
    except RuntimeError as e:
        # Handle CUDA OOM specifically
        err_lower = str(e).lower()
        if "out of memory" in err_lower or "cuda error" in err_lower:
            msg = (
                f"❌ OOM on {target_device}. Select another GPU or click "
                f"'Clear Memory & Load GPU0' to retry."
            )
            print(msg)
            model = None
            processor = None
            if torch.cuda.is_available():
                _clear_all_cuda()
            return msg, gr.update(interactive=False)
        raise


def ensure_model(target_device=None):
    """Ensure the model is loaded and returned (non-UI helper)."""
    global model, processor, device
    if target_device is None:
        target_device = device
    if model is None or device != target_device:
        load_model_on_device(target_device)
    return model, processor


def get_next_device(current_device: str):
    """Pick the next GPU (or CPU fallback)."""
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return "cpu"
    gpu_list = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if current_device not in gpu_list:
        return gpu_list[0]
    idx = gpu_list.index(current_device)
    return gpu_list[(idx + 1) % len(gpu_list)]

def separate_audio(video_file, mask_file):
    """
    Separate audio using video and mask files
    Returns: (target_audio_path, residual_audio_path, status, base)
    """
    try:
        # Initialize model if needed
        m, p = ensure_model()
        # Load video and mask
        print(f"Loading video: {video_file}")
        print(f"Loading mask: {mask_file}")
        frames, mask = load_input(video_file, mask_file)
        # Separate audio
        print("Separating audio...")
        output_dir = "outputs/gradio_sep"
        os.makedirs(output_dir, exist_ok=True)
        sep_audio(m, p, video_file, frames, mask, output_dir=output_dir)
        base = os.path.splitext(os.path.basename(video_file))[0]
        target_path = os.path.join(output_dir, f"{base}_target.wav")
        residual_path = os.path.join(output_dir, f"{base}_residual.wav")
        return target_path, residual_path, "✅ Audio separation completed successfully!", base
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        return None, None, error_msg, "output"

def clear_memory():
    """Clear CUDA memory"""
    global model, processor
    model = None
    processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return "Memory cleared successfully!"

# Create Gradio interface
with gr.Blocks(title="SAM Audio Separation Demo") as demo:
    gr.Markdown("""
    # 🎵 SAM Audio Separation Demo
    
    Separate audio from videos using SAM Audio with visual masking.
    
    **Instructions:**
    1. Upload or select a video file
    2. Upload or select the corresponding mask video file
    3. Click "Separate Audio" to process
    4. Download the separated target and residual audio files
    """)
    
    status_text = gr.Textbox(label="Status", interactive=False, value="Loading the SAM Audio model. Please wait...")
    device_dropdown = gr.Dropdown(
        choices=[c[0] for c in gpu_choices],
        value=gpu_choices[0][0],
        label="Model Device",
        info="Select GPU; CPU is a last-resort fallback (slow)."
    )
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input Files")
            video_input = gr.File(
                label="Video File (.mp4)",
                file_types=[".mp4", ".avi", ".mov"],
                type="filepath"
            )
            video_preview = gr.Video(label="Video Preview")
            mask_input = gr.File(
                label="Mask Video File (.mp4)",
                file_types=[".mp4", ".avi", ".mov"],
                type="filepath"
            )
            mask_preview = gr.Video(label="Mask Preview")
            
            separate_btn = gr.Button("🎵 Separate Audio", variant="primary", size="lg", interactive=False)
            clear_btn = gr.Button("🗑️ Clear Memory", size="sm")
            
            # Examples section
            gr.Markdown("### 📝 Try an Example")
            gr.Examples(
                examples=[
                    [
                        "/home/prj/play_sam3_audio/downloads/cUzeq_-hQ-o/cUzeq_-hQ-o_va_s720_f15.mp4",
                        "/home/prj/play_sam3_audio/outputs/sam3_mask_point_1766141782_mask.mp4"
                    ]
                ],
                inputs=[video_input, mask_input],
                label="Click to load example video and mask"
            )
            
            # Update video previews when files are loaded
            video_input.change(lambda f: f, inputs=video_input, outputs=video_preview)
            mask_input.change(lambda f: f, inputs=mask_input, outputs=mask_preview)
        
        with gr.Column():
            gr.Markdown("### Output Audio")
            def load_model_on_start(selected_device):
                status = "Loading the SAM Audio model. Please wait..."
                try:
                    status, button_state = load_model_on_device(selected_device)
                except Exception as e:
                    status = f"❌ Failed to load the model: {e}"
                    button_state = gr.update(interactive=False)
                return status, button_state, gr.update(value=selected_device)

            demo.load(
                fn=load_model_on_start,
                inputs=[device_dropdown],
                outputs=[status_text, separate_btn, device_dropdown]
            )
            load_btn = gr.Button("Load / Reload Model", variant="secondary")
            # Buttons for OOM recovery
            retry_gpu0_btn = gr.Button(
                "Clear Memory & Load GPU0",
                visible=torch.cuda.is_available(),
                size="sm"
            )
            other_gpu_btn = gr.Button(
                "Load on Next GPU",
                visible=torch.cuda.is_available() and torch.cuda.device_count() > 1,
                size="sm"
            )
            target_audio = gr.Audio(
                label="🎯 Target Audio (Foreground)",
                type="filepath"
            )
            residual_audio = gr.Audio(
                label="🔊 Residual Audio (Background)",
                type="filepath"
            )
            # New widget for blended video with separated audio
            gr.Markdown("### Blended Video with Separated Audio")
            blended_video = gr.Video(label="Blended Video + Separated Audio", interactive=False)
            blend_btn = gr.Button("Create Blended Video", variant="secondary")
    
    # Event handlers
    def separate_audio_and_return_base(video_file, mask_file):
        target, residual, status = None, None, ""
        base = "output"
        try:
            target, residual, status, base = separate_audio(video_file, mask_file)
        except Exception as e:
            status = str(e)
        return target, residual, status, base

    separate_btn.click(
        fn=separate_audio_and_return_base,
        inputs=[video_input, mask_input],
        outputs=[target_audio, residual_audio, status_text, gr.State()]
    )

    def ui_load_selected(selected_device):
        try:
            status, btn_state = load_model_on_device(selected_device)
            return status, btn_state, gr.update(value=selected_device)
        except Exception as e:
            return (
                f"❌ Failed to load the model: {e}",
                gr.update(interactive=False),
                gr.update(value=selected_device),
            )

    load_btn.click(
        fn=ui_load_selected,
        inputs=[device_dropdown],
        outputs=[status_text, separate_btn, device_dropdown]
    )

    def clear_and_load_gpu0():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        target = "cuda:0" if torch.cuda.is_available() else "cpu"
        return ui_load_selected(target)

    retry_gpu0_btn.click(
        fn=clear_and_load_gpu0,
        outputs=[status_text, separate_btn, device_dropdown]
    )

    def load_on_next_gpu(current_device):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            return ui_load_selected("cpu")
        next_dev = get_next_device(current_device)
        return ui_load_selected(next_dev)

    other_gpu_btn.click(
        fn=load_on_next_gpu,
        inputs=[device_dropdown],
        outputs=[status_text, separate_btn, device_dropdown]
    )

    def blend_and_audio(video_file, mask_file, target_audio, residual_audio):
        import os
        from utils.video_audio_blend import blend_video_with_mask_and_audio
        # Use the separated (target) audio for blending
        if not (video_file and mask_file and target_audio):
            return None
        base = os.path.splitext(os.path.basename(video_file))[0]
        output_path = f"outputs/gradio_sep/{base}_blended.mp4"
        blend_video_with_mask_and_audio(video_file, mask_file, target_audio, output_path)
        return output_path

    blend_btn.click(
        fn=blend_and_audio,
        inputs=[video_input, mask_input, target_audio],
        outputs=blended_video
    )
    
    clear_btn.click(
        fn=clear_memory,
        outputs=status_text
    )
    
    gr.Markdown("""
    ---
    ### Notes
    - This demo requires CUDA GPU
    - First run will download the SAM Audio model (~2GB)
    - Processing time depends on video length
    - Use "Clear Memory" if you encounter GPU memory issues
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False
    )
