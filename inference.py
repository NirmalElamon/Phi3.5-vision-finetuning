#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 13:47:29 2025

@author: nelamon
"""

from transformers import TextStreamer
from PIL import Image
import torch
import requests
from io import BytesIO
from decord import VideoReader, cpu
import argparse
import warnings
import pandas as pd
from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
import torch
import time 

model_path = "text_overlay_epoch_4"
model_base="microsoft/Phi-3-vision-128k-instruct"
device_map="cuda"
device="cuda"

disable_torch_init()

model_name = get_model_name_from_path(model_path)

use_flash_attn = False
max_new_tokens = 500
temperature = 0
repetition_penalty = 1.0


processor, model = load_pretrained_model(model_path = model_path, model_base=model_base, 
                                            model_name=model_name, device_map=device,
                                            device=device, use_flash_attn=use_flash_attn
)
generation_args = {
    "max_new_tokens": max_new_tokens,
    "temperature": temperature,
    "do_sample": True if temperature > 0 else False,
    "repetition_penalty": repetition_penalty,
}

# csv file which contains "url" column with image urls
df = pd.read_csv('textOverlay_test_set.csv')


generation_args = {
    "max_new_tokens": 10,
    "do_sample": True if temperature > 0 else False}


delay = 2
n = 0

#output file where the results will be stored
output_csv_path = 'predictions.csv'

# Initialize new columns for predictions
df['text_overlay_pred'] = None

with torch.no_grad():
    # Iterate through DataFrame and make predictions
    for index, row in df.iterrows():

        # URL of the image
        image_url = row['url']
        predictions = {}
        try:
            # Download the image from the URL
            images = []

            #response = requests.get(image_url)
            #image = Image.open(BytesIO(response.content)).convert("RGB")
            images.append(Image.open(requests.get(image_url, stream=True).raw))
        except Exception as e:
            print(f"Error processing image {image_url}: {e}")
            n += 1
            continue  # Skip this row and go to the next
        


        
        prompt_1 = """<|image_1|>\nThis is one still image of a vacation property. Does this image contain any artificial or graphical text overlay in it? This can be defined as when the image contains text that is not part of the property itself such as watermarks, timestamps, brand logos, or instructional text. Focus on text superimposed onto the image, not naturally integrated into the scene. Naturally occurring text in the scene should not be considered as text overlays. These include Hotel or building names, Billboards or signage, Posters, flyers, advertisements, License plates or any naturally visible text on objects (e.g., a book cover or product labels).Always return a yes for artificial text overlay in the image and no for images without any artificial text overlay."""
     

        messages = [
        {
            "role": "user",
            "content": prompt_1
        }
        ]

        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt, images, return_tensors="pt").to(device)
        

        print("about to perform inference")     
        with torch.inference_mode():
            generate_ids = model.generate(
                **inputs, 
                **generation_args,
                use_cache=True,
                eos_token_id=processor.tokenizer.eos_token_id
            )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response_1 = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        
        predictions['text_overlay_prompt_1'] = response_1


        del response_1  # Remove unused tensors
        torch.cuda.empty_cache()
        '''
        except Exception as e:
            print(f"Other error: {e}")
            predictions = {"text_overlay": "invalid"}  # Default
        '''
                
        # Update the DataFrame directly
        df.at[index, 'text_overlay_prompt_1'] = predictions.get('text_overlay_prompt_1', 'invalid')

        # Save the updated DataFrame to CSV after every iteration
        df.to_csv(output_csv_path, index=False)
        
        print(f"Iteration {n}")
        n += 1
        time.sleep(delay)



