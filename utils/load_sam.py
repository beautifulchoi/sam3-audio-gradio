#################################### For Image ####################################
# Load the model
def get_samaudio():
    try:
        from sam_audio import SAMAudio, SAMAudioProcessor
    except:
        print("sam audio is not installed")
        return 0
    try:
        model = SAMAudio.from_pretrained("facebook/sam-audio-large")
        processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-large")
        print("sam audio (large) is loaded well")
        return model, processor
    
    except:
        print("sam audio is not downloaded well. install it or request META to access the model")
        return 0

def get_sam(input_type='video', device_idx:int=1):
    try:
        from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
    except:
        print("sam3 is not installed")
        return 0   
    
    try:     
        bpe_path = "/home/prj/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        if input_type=='image':
            predictor = build_sam3_image_model(bpe_path=bpe_path, device="cuda", eval_mode=True) # return is nn.Module model
        elif input_type=='video':
            predictor = build_sam3_video_predictor(gpus_to_use=[device_idx]) # return is wrapper
        print("sam3 (large) is loaded well")
        return predictor
    except Exception as e:
        print("sam3 is not downloaded well. install it or request META to access the model")
        raise e
        #return 0

if __name__ == "__main__":
    get_samaudio()
    get_sam()