import gradio as gr
import torch
from diffusers import StableDiffusion3Pipeline

def img_gen(prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu" #selecting the device gpu or cpu
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16 if device == "cuda" else torch.float32, text_encoder_3 = None, tokenizer_3 = None) #a tensor using torch, and text_encoder and tokenizer is used to reduce resources used, but getting the same quality image.
    #pipe.enable_model_cpu_offload() #optimizes model, and moves it into CPU
    pipe.to(device) #load complete model in GPU

    image = pipe(
        prompt = prompt,
        negative_prompt = "ugly, watermark, low resolution, blurry",
        num_inference_steps= 50, #more number = better image quality using more resources
        height = 1024,
        width = 1024,
        guidance_scale = 9.0 #the measure of how much the model will follow the prompt, lesser number means the model will follow its ownself


    ).images[0]

    image.show()

print("Please enter the prompt of the image you want: ")
str = input()
img_gen(str)