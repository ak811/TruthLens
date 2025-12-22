import os
os.environ['PYTHONPATH'] = './:' + os.environ.get('PYTHONPATH', '')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import json
from tqdm import tqdm
from ChatUniVi.constants import *
from ChatUniVi.conversation import conv_templates, SeparatorStyle
from ChatUniVi.model.builder import load_pretrained_model
from ChatUniVi.utils import disable_torch_init
from ChatUniVi.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image

def process_single_image(image_path, model, tokenizer, image_processor, conv_mode="simple"):
    """Process a single image and return the model's response"""
    query ="Taking into account the lighting, texture, symmetry, and other features, describe the overall realism of the face. Does it show any signs of being digitally manipulated or generated?"
    temperature = 0.2
    top_p = None
    num_beams = 1
    
    try:
        image = Image.open(image_path)
        
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + query
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + query
            
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )
            
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
            
        return outputs.strip()
        
    except Exception as e:
        return f"Error: {str(e)}"

def process_dataset(dataset_path, model, tokenizer, image_processor):
    """Process all images in the dataset structure"""
    all_results = {}
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist.")
        return all_results

    for subfolder in tqdm(['fake1000', 'first1000'], desc="Processing folders"):
        subfolder_path = os.path.join(dataset_path, subfolder)
        if not os.path.exists(subfolder_path):
            print(f"Subfolder {subfolder_path} does not exist, skipping...")
            continue
            
        print(f"\nProcessing subfolder: {subfolder}")
        subfolder_results = {}
        
        image_files = [f for f in os.listdir(subfolder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        for file in tqdm(image_files, desc=f"Processing images in {subfolder}", leave=False):
            image_path = os.path.join(subfolder_path, file)
            response = process_single_image(
                image_path,
                model,
                tokenizer,
                image_processor
            )
            subfolder_results[os.path.abspath(image_path)] = response
        
        all_results[subfolder] = subfolder_results
        
        # Saving the intermediate results for each subfolder
        output_file = os.path.join(subfolder_path, f"realism_2_chat_univi_results_{subfolder}.json")
        with open(output_file, "w") as f:
            json.dump(subfolder_results, f, indent=4)
        print(f"Results for {subfolder} saved to {output_file}")
    
    # Saving the combined results
    combined_output_file = os.path.join(dataset_path, "realism_2_chat_univi_fake_results.json")
    with open(combined_output_file, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Combined results saved to {combined_output_file}")
    
    return all_results

def main():
    print("Initializing Chat-UniVi model...")

    model_path = "Chat-UniVi/Chat-UniVi"
    dataset_path = "your path"
    
    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_name = "ChatUniVi"
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    
    # Special tokens
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    
    # Loading vision tower
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor
    
    print("Starting image processing...")
    results = process_dataset(dataset_path, model, tokenizer, image_processor)
    print("Processing completed!")

if __name__ == '__main__':
    main()