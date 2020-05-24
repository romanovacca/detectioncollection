from src.core.utils.dataset import Dataset
from torch.utils.data import DataLoader

import torch
from torch.utils import data

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Parameters
params = {'batch_size': 8,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100

def collate_fn(batch):
    return tuple(zip(*batch))

# Generators
training_set = Dataset("/home/romano/PycharmProjects/detectioncollection/src/tests/data/train/images",
                       "/home/romano/PycharmProjects/detectioncollection/src/tests/data/train/annotations")
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset("/home/romano/PycharmProjects/detectioncollection/src/tests/data/test/images",
                         "/home/romano/PycharmProjects/detectioncollection/src/tests/data/test/annotations")
validation_generator = data.DataLoader(validation_set, **params)

def collate_fn(batch):
    return tuple(zip(*batch))

loader = DataLoader(training_set,
                    batch_size=2,
                    shuffle=True,
                    collate_fn=collate_fn)


def to_device(images, targets):
    images = [image.to(device) for image in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return images, targets

num_epochs= 5
learning_rate = 0
momentum =
weight_decay = 0
lr_step_size = 0
gamma = 0

losses = []
# Get parameters that have grad turned on (i.e. parameters that should be trained)
parameters = [p for p in model.parameters() if p.requires_grad]
# Create an optimizer that uses SGD (stochastic gradient descent) to train the parameters
optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
# Create a learning rate scheduler that decreases learning rate by gamma every lr_step_size epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma)


for epoch in range(num_epochs):
    #model.train()
    i = 0

    for imgs, annotations in loader:
        #imgs = list(img.to(device) for img in imgs)
        #annotations = list(zip(annotations["bounding_box"].to(device),annotations["label"]))
        #print(annotations)
        # print(imgs[0].shape)
        # print(annotations[0])

        images, targets = to_device(imgs, annotations)

        # Calculate the model's loss (i.e. how well it does on the current
        # image and target, with a lower loss being better)
        loss_dict = model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        # Zero any old/existing gradients on the model's parameters
        optimizer.zero_grad()
        # Compute gradients for each parameter based on the current loss calculation
        total_loss.backward()
        # Update model parameters from gradients: param -= learning_rate * param.grad
        optimizer.step()
