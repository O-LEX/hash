import gradio as gr
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from encorder import Encoder
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

to_pil = transforms.ToPILImage()

def process_image(img, num_steps, num_levels, features_per_level):
    if img is None:
        return None, "Please upload an image"
    
    img = Image.fromarray(img).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_tensor = transform(img)
    img_tensor = img_tensor.to(device)
    C, H, W = img_tensor.shape

    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, H, device=device),
        torch.linspace(0, 1, W, device=device),
        indexing="ij"
    )
    coords = torch.stack([xx, yy], dim=-1).view(-1, 2)
    target = img_tensor.permute(1, 2, 0).reshape(-1, 3)

    encoder = Encoder(input_dim=2, num_levels=num_levels, features_per_level=features_per_level).to(device)
    mlp = nn.Sequential(
        nn.Linear(num_levels * features_per_level, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 3),
        nn.Sigmoid()
    ).to(device)

    params = list(encoder.parameters()) + list(mlp.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        feats = encoder(coords)
        preds = mlp(feats)
        loss = F.mse_loss(preds, target)
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0 or step == num_steps - 1:
            with torch.no_grad():
                pred_img = preds.view(H, W, 3).detach().cpu()
                pred_img = pred_img.permute(2, 0, 1)
                pil_img = to_pil(pred_img)
            
            status = f"Step {step}/{num_steps} - Loss: {loss.item():.6f}"
            yield pil_img, status
    
    final_status = f"Training complete! Final loss: {loss.item():.6f}"
    yield pil_img, final_status

with gr.Blocks() as demo:
    gr.Markdown("# Hash Encoding Image Reconstruction")
    gr.Markdown("Upload an image and train a neural network to reconstruct it using hash encoding")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image")
            num_steps = gr.Slider(minimum=100, maximum=5000, value=1000, step=100, label="Training Steps")
            num_levels = gr.Slider(minimum=4, maximum=16, value=8, step=1, label="Number of Levels")
            features_per_level = gr.Slider(minimum=2, maximum=8, value=4, step=1, label="Features per Level")
            train_btn = gr.Button("Train")
        
        with gr.Column():
            output_image = gr.Image(label="Reconstructed Image")
            status_text = gr.Textbox(label="Status")
    
    train_btn.click(
        fn=process_image,
        inputs=[input_image, num_steps, num_levels, features_per_level],
        outputs=[output_image, status_text]
    )

if __name__ == "__main__":
    demo.launch()
