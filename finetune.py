#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 13:50:16 2025

@author: nelamon
"""

import os
import subprocess

# Define model name
MODEL_NAME = "microsoft/Phi-3.5-vision-instruct"
# MODEL_NAME = "microsoft/Phi-3-vision-128k-instruct"  # Uncomment to use this model

# Set environment variables
os.environ["PYTHONPATH"] = f"src:{os.environ.get('PYTHONPATH', '')}"

# Define training script and arguments
training_script = "src/training/train.py"
deepspeed_config = "scripts/zero3.json"

import sys

# Training parameters
args = [
    "deepspeed", training_script,
    "--deepspeed", deepspeed_config,
    "--model_id", MODEL_NAME,
    "--data_path", "Phi3-Vision-Finetune-main/data/output.json",
    "--image_folder", "text_overlay_finetune",
    "--tune_img_projector", "True",
    "--freeze_vision_tower", "True",
    "--freeze_llm", "False",
    "--bf16", "True",
    "--fp16", "False",
    "--disable_flash_attn2", "True",
    "--output_dir", "text_overlay_epoch_2",
    "--num_crops", "16",
    "--num_train_epochs", "2",
    "--per_device_train_batch_size", "1",
    "--gradient_accumulation_steps", "2",
    "--learning_rate", "2e-4",
    "--weight_decay", "0.",
    "--warmup_ratio", "0.03",
    "--lr_scheduler_type", "cosine",
    "--logging_steps", "1",
    "--tf32", "True",
    "--gradient_checkpointing", "True",
    "--report_to", "tensorboard",
    "--lazy_preprocess", "True",
    "--dataloader_num_workers", "2"
]


# Run the command and stream output in real-time
process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

# Print stdout and stderr line by line
for line in process.stdout:
    print(line.strip())  # Print each line as it comes
    sys.stdout.flush()  # Force immediate output in Databricks

for line in process.stderr:
    print("[STDERR]", line.strip())  # Print stderr messages
    sys.stderr.flush()  # Force immediate output in Databricks

# Wait for process completion
process.wait()
print("Training execution completed.")