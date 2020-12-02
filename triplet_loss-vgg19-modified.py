import os
import sys
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
from torch.utils import data
import torchvision.models as models


# initialize seeds
random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed(7)
torch.set_deterministic(True)
torch.backends.cudnn.benchmark = False
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"


# Save a snapshot of a given model
def save_checkpoint(save_path, model, optimizer):
    if save_path is None:
        return

    state_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    torch.save(state_dict, save_path)
    print(f'Model saved to: {save_path}')


# Load a snapshot of a given model
def load_checkpoint(model, save_path, optimizer):
    state_dict = torch.load(save_path, map_location='cuda:0')
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    print(f'Model loaded from: {save_path}')


def count_parameters(model):
    temp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model architecture:\n\n', model)
    print(f'\nThe model has {temp:,} trainable parameters')


# Select an anchor image, a positive image and a negative image for Triplet Loss
class FishDataset(data.Dataset):
    def __init__(self, categories, root_dir, set_size, transform=None):
        self.categories = categories
        self.root_dir = root_dir
        self.transform = transform
        self.set_size = set_size
        self.anchor = None
        self.positive = None
        self.negative = None

    def __len__(self):
        return self.set_size

    def __getitem__(self, idx):
        # Select the main fish
        fish_id = random.choice(self.categories)
        img_dir = self.root_dir + fish_id[0] + "/"

        anchor_image_name = random.choice(fish_id[1])
        positive_image_name = random.choice(fish_id[1])

        # Ensure we select a different image of the same
        # fish for the anchor image and positive image
        while positive_image_name == anchor_image_name:
            positive_image_name = random.choice(fish_id[1])

        self.anchor = Image.open(img_dir + '/' + anchor_image_name)
        self.positive = Image.open(img_dir + '/' + positive_image_name)

        # Select an image of a different fish to use as the negative image
        negative_fish_id = random.choice(self.categories)
        negative_image_name = random.choice(negative_fish_id[1])
        negative_img_dir = self.root_dir + negative_fish_id[0] + "/" + negative_image_name

        self.negative = Image.open(negative_img_dir)

        # print("Anchor:", self.anchor)
        # print("Positive:", self.positive)
        # print("Negative:", self.negative)

        if self.transform:
            self.anchor = self.transform(self.anchor)
            self.positive = self.transform(self.positive)
            self.negative = self.transform(self.negative)

        return self.anchor, self.positive, self.negative


# Creates n-way one shot learning evaluation
class NWayOneShotEvalSet(data.Dataset):
    def __init__(self, testing_fish_ids, training_fish_ids, root_training_dir, root_test_dir, set_size, num_way,
                 transform=None):
        self.testing_fish_ids = testing_fish_ids
        self.training_fish_ids = training_fish_ids
        self.root_training_dir = root_training_dir
        self.root_test_dir = root_test_dir
        self.set_size = set_size
        self.num_way = num_way
        self.label = np.random.randint(self.num_way)
        self.transform = transform
        self.test_main_img = None
        self.validation_set = []

    def __len__(self):
        return self.set_size

    def __getitem__(self, idx):
        self.label = np.random.randint(self.num_way)

        # Choose a reference image
        testing_fish_id = random.choice(self.testing_fish_ids)
        testing_img_dir = self.root_test_dir + testing_fish_id[0] + '/'
        image_name = random.choice(testing_fish_id[1])

        # Load reference image
        self.test_main_img = Image.open(testing_img_dir + '/' + image_name)

        # Perform transformations
        if self.transform:
            self.test_main_img = self.transform(self.test_main_img)

        self.validation_set = []

        # Find N number of distinct images, 1 from the same set as the reference
        for i in range(self.num_way):
            test_img = None
            test_character = None

            if i == self.label:
                for training_fish in self.training_fish_ids:
                    if testing_fish_id[0] in training_fish[0]:
                        test_category = testing_fish_id
                        test_img_name = random.choice(training_fish[1])
                        test_img = Image.open(self.root_training_dir + testing_fish_id[0] + "/" + test_img_name)
                        # print("Main image class:", testing_fish_id[0])
                        # print("Reference class:", testing_fish_id[0])
                        break
            else:
                # Select a random classname
                test_category = random.choice(self.training_fish_ids)
                test_character = random.choice(test_category[1])

                # Ensure the selected classname is not the same as the one we are validating
                while testing_fish_id[0] == test_category[0]:
                    test_category = random.choice(self.training_fish_ids)

                test_character = random.choice(test_category[1])

                # Load the validation image
                test_img = Image.open(self.root_training_dir + test_category[0] + '/' + test_character)

            # Perform transformations
            if self.transform:
                test_img = self.transform(test_img)

            # Add the chosen image to the N-way validation set
            if test_img is not None:
                self.validation_set.append((test_img, test_category[0]))
            else:
                print("No image selected")

        # print("Reference:", testing_fish_id[0])
        # print("Classnames:", [x[1] for x in self.validation_set])

        return (self.test_main_img, testing_fish_id[0]), self.validation_set, torch.from_numpy(
            np.array([self.label], dtype=int))


class SiameseNetwork(nn.Module):
    def __init__(self, vgg19_in):
        super(SiameseNetwork, self).__init__()

        self.vgg19 = vgg19_in
        self.convAdd1 = nn.Conv2d(512, 4096, 3)
        self.reluAdd = nn.ReLU(inplace=True)
       
    def forward(self, x):
        x = self.vgg19(x)
        
        x = self.convAdd1(x)
        x = self.reluAdd(x)
        
        x = torch.flatten(x, start_dim=1)
        return x


# training and validation after every epoch
def train(model, train_loader, val_loader, num_epochs, criterion, save_name, cuda_device, optimizer):
    best_val_loss = float("Inf")
    train_losses = []
    val_losses = []
    cur_step = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        print("Starting epoch " + str(epoch + 1))

        for anchor, positive, negative in tqdm(train_loader):
            # Forward
            anchor = anchor.to(cuda_device)
            positive = positive.to(cuda_device)
            negative = negative.to(cuda_device)
        
            x1 = model(anchor)
            x2 = model(positive)
            x3 = model(negative)

            # Triplet margin loss
            loss = criterion(x1, x2, x3)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        print('Epoch [{}/{}],Train Loss: {:.4f}'
              .format(epoch + 1, num_epochs, avg_train_loss))

        print("Min trained loss: ", min(train_losses))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            save_checkpoint(f'model_iteration_{epoch + 1}.pt', model, optimizer)
        if avg_train_loss <= min(train_losses):
            save_checkpoint(f'model_iteration_minimum_tl.pt', model, optimizer)

    print("Finished Training")

    return train_losses


def plot_comparisons(reference_tuple, images, predicted_index, correct):
    # images: [(PIL.Image, difference, classname), (PIL.image, difference)]
    # reference: (PIL.image, classname)
    fig = plt.figure(figsize=(13, 9))

    reference, reference_class = reference_tuple

    ax = []
    ax.append(plt.subplot(3, 5, 1))
    ax[-1].set_title(f'Reference Image\nClassname: {reference_class[0]}', fontsize=11)
    plt.imshow(transforms.ToPILImage()(reference.squeeze_(0)))

    image_count = 1

    for i, (image, difference, classname) in enumerate(images):

        # print("Image:", image)
        # print("Difference:", difference)
        # print("Classname:", classname)
        image_count += 1
        ax.append(plt.subplot(3, 5, image_count))
        ax[-1].set_title(f'Distance: {difference}\nClassname: {classname[0]}', fontsize=11)

        if i == predicted_index:
            ax[-1].set_title(f'Predicated Image\nDistance: {difference}\nClassname: {classname[0]}', fontsize=11)
        if classname[0] == reference_class[0]:
            ax[-1].set_title(f'Correct Image\nDistance: {difference}\nClassname: {classname[0]}', fontsize=11)
        if classname[0] == reference_class[0] and i == predicted_index:
            ax[-1].set_title(f'Correct prediction\nDistance: {difference}\nClassname: {classname[0]}', fontsize=11)
        plt.imshow(transforms.ToPILImage()(image.squeeze_(0)))

    output_path = f'{os.getcwd()}/comparison_outputs/comparison_{correct}.png'
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print("Saved comparison image to: ", output_path)


# evaluation metrics
def evaluate_model(model, test_loader, device):
    print('Starting Iteration')

    with torch.no_grad():
        model.eval()
        correct = 0
        count = 0

        for (main_image, main_classname), validation_set, label in test_loader:
            predicted_image = None
            # print("Main image:", main_image)
            # print("Main classname:", main_classname)
            main_image = main_image.to(device)
            smallest_difference = float('Inf')
            prediction_index = -1

            differences = []
            images = []

            for i, (test_image, test_class) in enumerate(validation_set):
                test_image = test_image.to(device)

                output1 = model(main_image)
                output2 = model(test_image)

                difference = torch.cdist(output1, output2)

                # images.append((test_image, f'{difference.item():.3f}', test_class))
                differences.append(difference)

                if difference < smallest_difference:
                    prediction_index = i
                    smallest_difference = difference

            if prediction_index == label.item():
                # print("Main image:", main_image)
                # print("Main classname:", main_classname)
                correct += 1

            # if (len(test_loader) % 10) == 0:
            #     plot_comparisons((main_image, main_classname), images, prediction_index, count)

            # print([f'{difference.item():.2f}' for difference in differences])

            count += 1

            if count % 10 == 0:
                print("Current Count is: {}".format(count))
                print('Accuracy on n way: {}'.format(correct / count))


vgg19 = models.vgg19_bn(pretrained=False)
params = list(vgg19.children())[:-1]
vgg19 = torch.nn.Sequential(*params)
torch.set_deterministic(False)

root_training_dir = f'{os.getcwd()}/training_data/'
root_validation_dir = f'{os.getcwd()}/testing_data/'
training_categories = [(folder, [image for image in os.listdir(root_training_dir + "/" + folder)]) for folder in
                       os.listdir(root_training_dir)]
validation_categories = [(folder, [image for image in os.listdir(root_validation_dir + "/" + folder)]) for folder in
                         os.listdir(root_validation_dir)]

# choose a training dataset size and further divide it into train and validation set 80:20
dataSize = 1000  # self-defined dataset size
TRAIN_PCT = 0.7  # percentage of entire dataset for training
train_size = int(dataSize * TRAIN_PCT)
val_size = dataSize - train_size

transformations = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

fishDataset = FishDataset(training_categories, root_training_dir, dataSize, transformations)

train_set, val_set = data.random_split(fishDataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, num_workers=32)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, num_workers=32, shuffle=True)

# creating the original network and couting the paramenters of different networks
cuda_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(cuda_device, torch.cuda.get_device_name(cuda_device))
siameseBaseLine = SiameseNetwork(vgg19).to(cuda_device)  # Net().to(cuda_device)

count_parameters(siameseBaseLine)

optimizer = optim.Adam(siameseBaseLine.parameters(), lr=1e-5)
num_epochs = 10
criterion = nn.TripletMarginLoss()

# load_checkpoint(siameseBaseLine, f'./model_iteration_minimum_tl.pt', optimizer)

train_losses = train(
     model=siameseBaseLine,
     train_loader=train_loader,
     val_loader=val_loader,
     num_epochs=num_epochs,
     criterion=criterion,
     save_name='',
     cuda_device=cuda_device,
     optimizer=optimizer
 )

# create the test set for final testing
set_size = 1000
num_way = 10

test_set = NWayOneShotEvalSet(
    testing_fish_ids=validation_categories,
    training_fish_ids=training_categories,
    root_training_dir=root_training_dir,
    root_test_dir=root_validation_dir,
    set_size=set_size,
    num_way=num_way,
    transform=transformations
)

# save_checkpoint(f'{os.getcwd()}/model_final.pt', siameseBaseLine, optimizer)
# optimizer = optim.Adam(model.parameters(), lr=1e-5)
model = SiameseNetwork(vgg19).to(cuda_device)
# model = models.resnet18(pretrained=True).to(cuda_device)
load_checkpoint(model, f'./accuracy_29/model_iteration_minimum_tl_extra_conv.pt', optimizer)
test_loader = torch.utils.data.DataLoader(test_set, num_workers=32, shuffle=True)
#model.eval()
evaluate_model(model, test_loader, cuda_device)

