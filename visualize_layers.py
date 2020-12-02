import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt

class SiameseNetwork(nn.Module):
    def __init__(self, vgg19):
        super(SiameseNetwork, self).__init__()
        self.vgg19 = vgg19

    def forward(self, x):
        x = self.vgg19(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.linalg.norm(x, ord=2, dim=1, keepdim=True)
        return x

# Load a snapshot of a given model
def load_checkpoint(model, save_path, optimizer):
    state_dict = torch.load(save_path, map_location='cuda:0')
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    print(f'Model loaded from: {save_path}')

CWD = os.getcwd()
vgg19 = models.vgg19(pretrained=False)
vgg19 = nn.Sequential(*list(vgg19.children()))[:-1]
model = SiameseNetwork(vgg19)
optimizer = optim.Adam(model.parameters(), lr=5e-5)
load_checkpoint(model, f'{os.getcwd()}/model_iteration_50.pt', optimizer)

conv2d_layers = []
model_weights = []

# Extract all convolutional layers
for sequential in model.children():
    for child in sequential.children():
        if type(child) == torch.nn.Conv2d:
            conv2d_layers.push(child)
            model_weights.append(child.weight)

        for nested_child in child.children():
            if type(nested_child) == torch.nn.Conv2d:
                conv2d_layers.append(nested_child)
                model_weights.append(nested_child.weight)
        
        for weight, conv in zip(model_weights, conv2d_layers):
            print(f"CONV: {conv} ====> SHAPE: {weight.shape}")

plt.figure(figsize=(20, 17))

for i, filter in enumerate(model_weights[0]):
    plt.subplot(8, 8, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
    plt.imshow(filter[0, :, :].detach(), cmap='gray')
    plt.axis('off')
    plt.savefig(f'{CWD}/visualizations/filter.png')

plt.show()

# read and visualize an image
img = cv.imread(f'{CWD}/original_dataset/65335_l_2.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

# define the transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
img = np.array(img)
# apply the transforms
img = transform(img)
print(img.size())
# unsqueeze to add a batch dimension
img = img.unsqueeze(0)
print(img.size())


# pass the image through all the layers
results = [conv2d_layers[0](img)]
for i in range(1, len(conv2d_layers)):
    # pass the result from the last layer to the next layer
    results.append(conv2d_layers[i](results[-1]))
# make a copy of the `results`
outputs = results

# visualize 64 features from each layer 
# (although there are more feature maps in the upper layers)
for num_layer in range(len(outputs)):
    plt.figure(figsize=(30, 30))
    layer_viz = outputs[num_layer][0, :, :, :]
    layer_viz = layer_viz.data
    print(layer_viz.size())
    for i, filter in enumerate(layer_viz):
        if i == 64: # we will visualize only 8x8 blocks from each layer
            break
        plt.subplot(8, 8, i + 1)
        plt.imshow(filter, cmap='gray')
        plt.axis("off")
    print(f"Saving layer {num_layer} feature maps...")
    plt.savefig(f"{CWD}/visualizations/layer_{num_layer}.png")
    # plt.show()
    plt.close()
