import gradio as gr
import numpy as np
import random
import torch
from diffusers import SanaSprintPipeline
import gc # Import garbage collector

# Determine device and dtype
if torch.cuda.is_available():
    device = "cuda"
    # Check if bfloat16 is supported on the CUDA device
    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        print(f"Using device: {device} with dtype: {dtype}")
    else:
        dtype = torch.float16
        print(f"CUDA device detected, but bfloat16 not supported. Using device: {device} with dtype: {dtype}")
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16 # Use float16 for MPS
    print(f"Using device: {device} with dtype: {dtype}")
else:
    device = "cpu"
    dtype = torch.float32 # Use float32 for CPU
    print(f"Using device: {device} with dtype: {dtype}")

# Initialize pipelines to None
pipe = None
pipe2 = None
current_model_on_device = None # Track which model is on the GPU

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

def infer(prompt, model_size, seed=42, randomize_seed=False, width=1024, height=1024, guidance_scale=4.5, num_inference_steps=2):
    global pipe, pipe2, current_model_on_device # Access global variables

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)

    # Determine which pipeline is needed
    target_pipe_name = "0.6B" if model_size == "0.6B" else "1.6B"
    selected_pipe = None

    # Check if the correct model is already loaded and on the device
    if current_model_on_device == target_pipe_name:
        selected_pipe = pipe if target_pipe_name == "0.6B" else pipe2
    else:
        # Free memory from the old model if it exists and is on the device
        if current_model_on_device == "0.6B" and pipe is not None:
            print(f"Moving model 0.6B from {device} to cpu")
            pipe.to("cpu")
            pipe = None # Allow garbage collection
        elif current_model_on_device == "1.6B" and pipe2 is not None:
            print(f"Moving model 1.6B from {device} to cpu")
            pipe2.to("cpu")
            pipe2 = None # Allow garbage collection

        # Explicitly clear cache if on MPS
        if device == "mps":
            torch.mps.empty_cache()
        gc.collect() # Run garbage collection

        # Load the target model if it's not loaded
        print(f"Loading model {target_pipe_name}...")
        if target_pipe_name == "0.6B":
            pipe = SanaSprintPipeline.from_pretrained(
                "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
                torch_dtype=dtype
            )
            selected_pipe = pipe
        else: # target_pipe_name == "1.6B"
            pipe2 = SanaSprintPipeline.from_pretrained(
                "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
                torch_dtype=dtype
            )
            selected_pipe = pipe2

        # Move the newly loaded or selected model to the device
        print(f"Moving model {target_pipe_name} to {device}...")
        selected_pipe.to(device)
        current_model_on_device = target_pipe_name # Update tracker

    # Perform inference
    img = selected_pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
            output_type="pil"
    )
    print(img)
    return img.images[0], seed
    
examples = [
    ["a tiny astronaut hatching from an egg on the moon", "1.6B"],
    ["🐶 Wearing 🕶 flying on the 🌈", "1.6B"],
    ["an anime illustration of a wiener schnitzel", "0.6B"],
    ["a photorealistic landscape of mountains at sunset", "0.6B"],
]

css="""
#col-container {
    margin: 0 auto;
    max-width: 520px;
}
"""

with gr.Blocks(css=css) as demo:
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""# Sana Sprint""")
        gr.Markdown("Demo for the real-time [Sana Sprint](https://huggingface.co/collections/Efficient-Large-Model/sana-sprint-67d6810d65235085b3b17c76) model")
        with gr.Row():
            
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            
            run_button = gr.Button("Run", scale=0)
        
        result = gr.Image(label="Result", show_label=False)
        
        # Set default model based on device
        default_model_size = "0.6B" if device == "mps" else "1.6B"
        print(f"Setting default model size to: {default_model_size}")

        model_size = gr.Radio(
            label="Model Size",
            choices=["0.6B", "1.6B"],
            value=default_model_size, # Use conditional default
            interactive=True
        )
        
        with gr.Accordion("Advanced Settings", open=False):
            
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            
            with gr.Row():
                
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )
                
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )
            
            with gr.Row():

                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1,
                    maximum=15,
                    step=0.1,
                    value=4.5,
                )
  
                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=2,
                )
        
        gr.Examples(
            examples = examples,
            fn = infer,
            inputs = [prompt, model_size],  # Add model_size to inputs
            outputs = [result, seed],
            cache_examples="lazy"
        )

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn = infer,
        inputs = [prompt, model_size, seed, randomize_seed, width, height, guidance_scale, num_inference_steps],  # Add model_size to inputs
        outputs = [result, seed]
    )

demo.launch()