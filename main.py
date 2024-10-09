import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image

from vgg import VGG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG().to(device).eval()
im_size = 356

total_steps = 2000
learning_rate = 0.001
alpha = 1
beta = 0.01

loader = transforms.Compose(
    [transforms.Resize((im_size, im_size)), transforms.ToTensor()]
)


def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)


original_image = load_image("original_images/original_image.jpg")
style_image = load_image("style/style_1.jpeg")

generated_image = original_image.clone().requires_grad_(True)
optimizer = optim.Adam([generated_image], lr=learning_rate)


for step in range(total_steps):
    # Obtain the convolution features in specifically chosen layers
    generated_features = model(generated_image)
    original_image_features = model(original_image)
    style_features = model(style_image)

    style_loss = original_loss = 0

    for gen_feature, orig_feature, style_feature in zip(
        generated_features, original_image_features, style_features
    ):
        batch_size, channel, height, width = gen_feature.shape

        original_loss += torch.mean((gen_feature - orig_feature) ** 2)

        G = gen_feature.view(channel, height * width).mm(
            gen_feature.view(channel, height * width).t()
        )
        A = style_feature.view(channel, height * width).mm(
            style_feature.view(channel, height * width).t()
        )

        style_loss += torch.mean((G - A) ** 2)

    total_loss = alpha * original_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(total_loss)
        save_image(generated_image, "generated_images/generated.png")
