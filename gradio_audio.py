import gradio as gr
import torch
import os
from audio_sep import load_input, sep_audio
from sam_audio import SAMAudio, SAMAudioProcessor

# Global model and processor
model = None
processor = None

def initialize_model():
    """Initialize SAM Audio model and processor"""
    global model, processor
    if model is None:
        print("Loading SAM Audio model...")
        model = SAMAudio.from_pretrained("facebook/sam-audio-large")
        processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-large")
        model = model.eval().cuda()
        print("Model loaded successfully!")
    return model, processor

def separate_audio(video_file, mask_file):
    """
    Separate audio using video and mask files
    Returns: (target_audio_path, residual_audio_path, status, base)
    """
    try:
        # Initialize model if needed
        m, p = initialize_model()
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
            # On app load, load the model and update status, enable button
            def load_model_on_start():
                status = "Loading the SAM Audio model. Please wait..."
                try:
                    initialize_model()
                    status = "✅ The SAM Audio model is loaded. Ready."
                except Exception as e:
                    status = f"❌ Failed to load the model: {e}"
                return status, gr.update(interactive=True)

            demo.load(
                fn=load_model_on_start,
                inputs=[],
                outputs=[status_text, separate_btn]
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

    def blend_and_audio(video_file, mask_file, target_audio):
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
