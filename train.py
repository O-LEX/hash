from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from encorder import Encoder

path = 'example.jpg'

to_pil = transforms.ToPILImage()
def save_img(img_tensor, step): # [H, W, C]
    img_tensor = img_tensor.permute(2, 0, 1) # [C, H, W]
    pil_img = to_pil(img_tensor)
    save_path = f"output/output_step_{step:04d}.png"
    pil_img.save(save_path)
    # print(f"Image saved to {save_path}")

device = "cuda" if torch.cuda.is_available() else "cpu"
img = Image.open(path).convert("RGB")
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
coords = torch.stack([xx, yy], dim=-1).view(-1, 2) # [H*W, 2]

target = img_tensor.permute(1, 2, 0).reshape(-1, 3) # [H*W, 3]

encoder = Encoder(input_dim=2, num_levels=8, features_per_level=4).to(device)
mlp = nn.Sequential(
    nn.Linear(8 * 4, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 3),
    nn.Sigmoid()
).to(device)

params = list(encoder.parameters()) + list(mlp.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)


for step in range(1000):
    optimizer.zero_grad()
    feats = encoder(coords)
    preds = mlp(feats)
    loss = F.mse_loss(preds, target)
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}: loss={loss.item():.6f}")
        with torch.no_grad():
            pred_img = preds.view(H, W, 3).detach().cpu()
            save_img(pred_img, step)