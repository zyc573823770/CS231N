import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import matplotlib.pyplot as plot
from torch.utils.tensorboard import SummaryWriter

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

NUM_TRAIN = 49000

LR = 1e-3
BATCH_SIZE = 64

# The torchvision.transforms package provides tools for preprocessing data
# and for performing data augmentation; here we set up a transform to
# preprocess the data by subtracting the mean RGB value and dividing by the
# standard deviation of each RGB value; we've hardcoded the mean and std.
transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

# We set up a Dataset object for each split (train / val / test); Datasets load
# training examples one at a time, so we wrap each Dataset in a DataLoader which
# iterates through the Dataset and forms minibatches. We divide the CIFAR-10
# training set into train and val sets by passing a Sampler object to the
# DataLoader telling how it should sample from the underlying Dataset.
cifar10_train = dset.CIFAR10('D:\\Desktop\\Outside\\CS231n(2017)\\assgn\\spring1819_assignment2\\assignment2\\cs231n\\datasets', train=True, download=True,
                             transform=transform)                 
loader_train = DataLoader(cifar10_train, batch_size=BATCH_SIZE, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

cifar10_val = dset.CIFAR10('D:\\Desktop\\Outside\\CS231n(2017)\\assgn\\spring1819_assignment2\\assignment2\\cs231n\\datasets', train=True, download=True,
                           transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=BATCH_SIZE, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))                        

cifar10_test = dset.CIFAR10('D:\\Desktop\\Outside\\CS231n(2017)\\assgn\\spring1819_assignment2\\assignment2\\cs231n\\datasets', train=False, download=True, 
                            transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=BATCH_SIZE)

USE_GPU = True

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss
print_every = 100

print('using device:', device)

def check_accuracy_part34(loader, model, label):
    print('Checking accuracy on %s set'%(label))   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('%s Got %d / %d correct (%.2f)' % (label, num_correct, num_samples, 100 * acc))
    return acc

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

from tqdm import tqdm


Writer = SummaryWriter('D:\\Desktop\\Outside\\CS231n(2017)\\assgn\\spring1819_assignment2\\assignment2\\run')
print_every = 100
def train_part34(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    al_loss_history = []
    al_val_acc_history = []
    al_train_acc_history = []
    ind = 0
    for e in range(epochs):
        loss_history = []
        val_acc_history = []
        train_acc_history = []
        for t, (x, y) in enumerate(loader_train):
            ind += 1
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)
            loss_history.append(loss)
            al_loss_history.append(loss)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                val_acc = check_accuracy_part34(loader_val, model, 'val')
                Writer.add_scalar('epoch%d loss'%(e), loss, t)
                Writer.add_scalar('epoch%d val acc'%(e), val_acc, t)
                Writer.add_scalar('all val acc', val_acc, ind)
                Writer.add_scalar('all loss', loss, ind)
                # train_acc = check_accuracy_part34(loader_train, model, 'train')
                # val_acc_history.append(val_acc)
                # train_acc_history.append(train_acc)
                # al_train_acc_history.append(train_acc)
                # al_val_acc_history.append(val_acc)
        # plot.figure(e)
        # plot.subplot(3,1,1)
        # plot.plot(loss_history)
        # plot.xlabel('iter')
        # plot.ylabel('loss')
        # plot.title('loss')
        # plot.subplot(3,1,2)
        # plot.plot(train_acc_history)
        # plot.xlabel('iter')
        # plot.ylabel('acc')
        # plot.title('train acc')
        # plot.subplot(3,1,3)
        # plot.plot(val_acc_history)
        # plot.xlabel('iter')
        # plot.ylabel('acc')
        # plot.title('val_acc')
        # plot.show(block=False)
    
    # plot.figure()
    # plot.subplot(3,1,1)
    # plot.plot(al_loss_history)
    # plot.xlabel('iter')
    # plot.ylabel('loss')
    # plot.title('loss')
    # plot.subplot(3,1,2)
    # plot.plot(al_train_acc_history)
    # plot.xlabel('iter')
    # plot.ylabel('acc')
    # plot.title('train acc')
    # plot.subplot(3,1,3)
    # plot.plot(al_val_acc_history)
    # plot.xlabel('iter')
    # plot.ylabel('acc')
    # plot.title('val_acc')
    # plot.show(block=False)
        

################################################################################
# TODO:                                                                        #         
# Experiment with any architectures, optimizers, and hyperparameters.          #
# Achieve AT LEAST 70% accuracy on the *validation set* within 10 epochs.      #
#                                                                              #
# Note that you can use the check_accuracy function to evaluate on either      #
# the test set or the validation set, by passing either loader_test or         #
# loader_val as the second argument to check_accuracy. You should not touch    #
# the test set until you have finished your architecture and  hyperparameter   #
# tuning, and only run the test set once at the end to report a final value.   #
################################################################################
model = None
optimizer = None

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
from collections import OrderedDict
model = nn.Sequential(
    OrderedDict([
    ('conv1', nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)),
    ('bn1', nn.BatchNorm2d(32)),
    ('relu1', nn.ReLU()),
    ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),
    ('conv2', nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
    ('bn2', nn.BatchNorm2d(32)),
    ('relu2', nn.ReLU()),
    ('flatten', Flatten()),
    ('fc1', nn.Linear(32*16*16, 4000)),
    ('relu3', nn.ReLU()),
    ('fc2', nn.Linear(4000, 256)),
    ('relu4', nn.ReLU()),
    ('fc3', nn.Linear(256, 10))
    ])    
    # nn.Conv2d(3, 128, kernel_size=5, stride=1, padding=2),
    # nn.ReLU(),
    # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
    # nn.ReLU(),
    # Flatten(),
    # nn.Linear(256*32*32, 10),
)
# model.to(device)

optimizer = optim.Adam(model.parameters(), LR, (0.9, 0.999))

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
################################################################################
#                                 END OF YOUR CODE                             
################################################################################

# You should get at least 70% accuracy
train_part34(model, optimizer, epochs=10)                

best_model = model
check_accuracy_part34(loader_test, best_model, 'test')