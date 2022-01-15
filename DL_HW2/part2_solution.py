# Don't erase the template code, except "Your code here" comments.

import torch
# Your code here...
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
# delete at the end
from IPython import display
from torch import nn

# import wandb
# import math
# import random

torch.manual_seed(0)

def get_dataloader(path, kind):
    """
    Return dataloader for a `kind` split of Tiny ImageNet.
    If `kind` is 'val', the dataloader should be deterministic.
    
    path:
        `str`
        Path to the dataset root - a directory which contains 'train' and 'val' folders.
    kind:
        `str`
        'train' or 'val'
        
    return:
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        For each batch, should yield a tuple `(preprocessed_images, labels)` where
        `preprocessed_images` is a proper input for `predict()` and `labels` is a
        `torch.int64` tensor of shape `(batch_size,)` with ground truth class labels.
    """
    # Your code here

    if kind == 'train':
        shuffle = True
        batch_size = 512
        pin_memory = True
        num_workers = 2
        transformation = transforms.Compose([
                                            transforms.RandomRotation(degrees = (5,10)),
                                            transforms.RandomHorizontalFlip(p = 0.25),
                                            torchvision.transforms.RandomVerticalFlip(p = 0.25),
                                            transforms.RandomAffine(degrees = 10, scale = (0.9, 1.1), shear = 10),
                                            transforms.ColorJitter(brightness = 0.05, contrast = 0.05, saturation = 0.05, hue = 0.05),
                                            transforms.RandomGrayscale(p = 0.05),
                                            transforms.ToTensor()
                                            ])

    if kind == 'val':
        shuffle = True
        batch_size = 512
        pin_memory = True
        num_workers = 2
        transformation = transforms.Compose([
                                            transforms.ToTensor()
                                            ])

    kind_data = torchvision.datasets.ImageFolder('tiny-imagenet-200/' + kind,transform = transformation)
    kind_dataloader = DataLoader(
                                kind_data,
                                batch_size = batch_size,
                                shuffle = shuffle,
                                pin_memory = pin_memory,
                                num_workers = num_workers
                                )
    print()                            
    print(kind + " dataloader has " + str(len(kind_dataloader)) + " batches")
    
    if kind == 'train':
        fig, axarr = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle("Several examples",fontsize=16)
        for idx, ax in enumerate(axarr.ravel()):
            for x,_ in kind_dataloader:
                buff = np.zeros([64,64,3])
                for i in range(3):
                    buff[:,:,i] = x[0,i,:,:]
                ax.imshow(buff)
                ax.axis('off')
                break
        print()   
        for x,y in kind_dataloader:
            print('Train data: ',x.shape)
            print('Train label:',y.shape)
            print() 
            print('Data in batch:',x.shape[0])
            print('RGB channels: ',x.shape[1])
            print('Height:       ',x.shape[2])
            print('Width:        ',x.shape[3])
            break
    else:
        None
    
    return kind_dataloader

def get_model():
    """
    Create neural net object, initialize it with raw weights, upload it to GPU.
    
    return:
    model:
        `torch.nn.Module`
    """
    # Your code here
    class NeuralNet(torch.nn.Module):
        def __init__(self, n_hidden_neurons):
            super(NeuralNet,self).__init__()

            self.bn0 = torch.nn.BatchNorm2d(3)
            # ======================================================================================================================
            self.conv1 = torch.nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1,bias = False)
            self.fa1 = torch.nn.ReLU(inplace=True)
            self.bn1 = torch.nn.BatchNorm2d(64)
            self.mp1 = torch.nn.MaxPool2d(kernel_size = 3,padding=1,stride=2)
            # ======================================================================================================================
            self.conv2 = torch.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1,bias = False)
            self.fa2 = torch.nn.ReLU(inplace=True)
            self.bn2 = torch.nn.BatchNorm2d(128)
            self.mp2 = torch.nn.MaxPool2d(kernel_size = 3,padding=1,stride=2)
            # ======================================================================================================================
            self.conv3 = torch.nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1,bias = False)
            self.fa3 = torch.nn.ReLU(inplace=True)
            self.bn3 = torch.nn.BatchNorm2d(256)
            self.mp3 = torch.nn.MaxPool2d(kernel_size = 2,padding=1,stride=2)
            # ======================================================================================================================
            self.conv4 = torch.nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1,bias = False)
            self.fa4 = torch.nn.ReLU(inplace=True)
            self.bn4 = torch.nn.BatchNorm2d(512)
            self.mp4 = torch.nn.MaxPool2d(kernel_size = 2,padding=1,stride=2)
            # ======================================================================================================================
            # self.conv5 = torch.nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3, stride = 2, padding = 1,bias = False)
            # self.fa5 = torch.nn.ReLU(inplace=True)
            # self.bn5 = torch.nn.BatchNorm2d(1024)
            # ======================================================================================================================
            # self.conv6 = torch.nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1,bias = False)
            # self.bn6 = torch.nn.BatchNorm2d(512)
            # self.fa6 = torch.nn.ReLU(inplace=True)
            # ======================================================================================================================
            # self.ap7 = torch.nn.AvgPool2d(kernel_size=3,padding=1,stride=2)
            self.fla7 = torch.nn.Flatten()
            self.fc7 = nn.Linear(12800, 7200)
            self.fa7 = torch.nn.ReLU(inplace=True)
            self.bn7 = torch.nn.BatchNorm1d(7200)
            self.do7 = torch.nn.Dropout(0.3)
            # ======================================================================================================================
            self.fc8 = nn.Linear(7200, 3600)
            self.fa8 = torch.nn.ReLU(inplace=True)
            self.bn8 = torch.nn.BatchNorm1d(3600)
            self.do8 = torch.nn.Dropout(0.3)
            # ======================================================================================================================
            self.fc9 = nn.Linear(3600, 200)
            self.lsm9 = torch.nn.LogSoftmax(dim=1)
            # ===============================

        def forward(self, x):
            x = self.bn0(x)
            # ===============================
            x = self.conv1(x)
            # print('after conv1 = ',x.shape)
            x = self.fa1(x)
            x = self.bn1(x)
            x = self.mp1(x)
            # ===============================
            x = self.conv2(x)
            x = self.fa2(x)
            x = self.bn2(x)
            x = self.mp2(x)
            # ===============================
            x = self.conv3(x)
            x = self.fa3(x)
            x = self.bn3(x)
            x = self.mp3(x)
            # ===============================
            x = self.conv4(x)
            x = self.fa4(x)
            x = self.bn4(x)
            x = self.mp4(x)
            # ===============================
            # x = self.conv5(x)
            # x = self.fa5(x)
            # x = self.bn5(x)
            # ===============================
            # x = self.conv6(x)
            # x = self.bn6(x)
            # x = self.fa6(x)
            # ===============================
            # x = self.ap7(x)
            x = self.fla7(x)
            # print('after fla7 = ',x.shape)
            x = self.fc7(x)
            x = self.fa7(x)
            x = self.bn7(x)
            x = self.do7(x)
            # ===============================
            x = self.fc8(x)
            x = self.fa8(x)
            x = self.bn8(x)
            x = self.do8(x)
            # ===============================
            x = self.fc9(x)
            x = self.lsm9(x)
            # ===============================
            return x

    # set up device
    use_cuda = True
    available = torch.cuda.is_available()
    print()
    if use_cuda:
        if available:
            device = torch.device("cuda")
            dtype = torch.cuda.FloatTensor
            print("Using GPU CUDA")
        else:
            device = torch.device("cpu")
            dtype = torch.FloatTensor
            print("Using CPU, CUDA is not available")

    if not use_cuda:
        if available:
            device = torch.device("cpu")
            dtype = torch.cuda.FloatTensor
            print("Using CPU, but CUDA is available")
        else:
            device = torch.device("cpu")
            dtype = torch.FloatTensor
            print("Using CPU, CUDA is not available")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = NeuralNet(10)
    model = model.to(device)
    return model

def get_optimizer(model):
    """
    Create an optimizer object for `model`, tuned for `train_on_tinyimagenet()`.
    
    return:
    optimizer:
        `torch.optim.Optimizer`
    """
    # Your code here
    learning_rate = 1.0e-3
    l2reg = 1.0e-4
    # optimizer = torch.optim.Adam(
    #                             model.parameters(),
    #                             lr=learning_rate,
    #                             eps=1e-08,
    #                             weight_decay=l2reg
                                # )

    optimizer = torch.optim.SGD(
                                    model.parameters(),
                                    lr=learning_rate,
                                    momentum=0.9
                                    )            
    return optimizer

def predict(model, batch):
    """
    model:
        `torch.nn.Module`
        The neural net, as defined by `get_model()`.
    batch:
        unspecified
        A batch of Tiny ImageNet images, as yielded by `get_dataloader(..., 'val')`
        (with same preprocessing and device).
    
    return:
    prediction:
        `torch.tensor`, shape == (N, 200), dtype == `torch.float32`
        The scores of each input image to belong to each of the dataset classes.
        Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
        belong to `j`-th class.
        These scores can be 0..1 probabilities, but for better numerical stability
        they can also be raw class scores after the last (usually linear) layer,
        i.e. BEFORE softmax.
    """
    # Your code here
    if batch.device.type == 'cpu':
        prediction = model.forward(batch.to(0))
    else:
        prediction = model.forward(batch)
    return prediction

def validate(dataloader, model):
    """
    Run `model` through all samples in `dataloader`, compute accuracy and loss.
    
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        See `get_dataloader()`.
    model:
        `torch.nn.Module`
        See `get_model()`.

    return:
    accuracy:
        `float`
        The fraction of samples from `dataloader` correctly classified by `model`
        (top-1 accuracy). `0.0 <= accuracy <= 1.0`
    loss:
        `float`
        Average loss over all `dataloader` samples.
    """
    # Your code here
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    total, correct = 0, 0
    for images_batch, labels_batch in tqdm(dataloader):
        with torch.no_grad():
            # move data to the GPU
            images_batch = images_batch.to(device)
            labels_batch = labels_batch.to(device)
            # get probabilities
            probs = model(images_batch)
            # choose max probabilities lables
            predictions = probs.max(axis = 1)[1]
            # compare lables and predictions
            correct += (predictions == labels_batch).sum().item()
            # total amount of images
            total += len(labels_batch)
            # calculate loss value
            val_loss = criterion(probs, labels_batch)

    # total amount of images
    val_accuracy = correct / total
    return val_accuracy, val_loss

def train_on_tinyimagenet(train_dataloader, val_dataloader, model, optimizer):
    """
    Train `model` on `train_dataloader` using `optimizer`. Use best-accuracy settings.
    
    train_dataloader:
    val_dataloader:
        See `get_dataloader()`.
    model:
        See `get_model()`.
    optimizer:
        See `get_optimizer()`.
    """
    # Your code here

    # wandb.init(project="HW2_2", config={
    # "architecture": "CNN",
    # "dataset": "Tiny-imagenet-200"})
    # config = wandb.config

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, (10, 30), gamma = 0.15)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0005, max_lr=0.005)

    n_epochs = 50

    seed_value = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    criterion = torch.nn.CrossEntropyLoss()

    train_loss_history = []
    train_acc_history = []

    val_loss_history = []
    val_acc_history = []
    max_val_accuracy_value = 0
    val_acc_max_epoch = 0

    best_accuracy = 0

    for EPOCH in range(n_epochs):

        model.train()
        train_acc_epoch = 0
        train_loss_epoch = 0
        for batches, (images_batch, labels_batch) in enumerate(tqdm(train_dataloader)):
            # move data to the GPU
            images_batch = images_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            # zero gradient
            optimizer.zero_grad()
            # make prediction
            predictions = predict(model,images_batch)
            # calculate loss
            loss_batch = criterion(predictions, labels_batch)
            # back propagation
            loss_batch.backward()
            # make optimizer step
            optimizer.step()

            train_loss_epoch += loss_batch.item()
            if scheduler is not None:
                scheduler.step()

        # save loss and accuracy
        train_loss_epoch = train_loss_epoch / batches
        train_loss_history.append(train_loss_epoch)

        model.eval()
        val_accuracy = get_accuracy(model, val_dataloader, device)
        val_acc_history.append(val_accuracy)
        train_accuracy = get_accuracy(model, train_dataloader, device)
        train_acc_history.append(train_accuracy)

        # display.clear_output(wait=True)
        # f, axes = plt.subplots(1, 2, figsize=(15, 3))
        # axes[0].set_title('Training loss')
        # axes[0].plot(train_loss_history)
        # axes[1].set_title('Validation accuracy')
        # axes[1].plot(val_acc_history)
        # axes[1].plot(train_acc_history,'r')
        # plt.tight_layout()
        # plt.show()

        print('Epoch number:',EPOCH)
        print('Train loss',train_loss_epoch)
        print('Train acc,% = ',train_accuracy * 100)
        print('Val acc,% = ',val_accuracy * 100)

        # save the model for the best parameters
        chckpnt_path = './checkpoint.pth'
        LOSS = train_loss_epoch
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_res_epoch = EPOCH
            torch.save({
                'epoch': EPOCH,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': LOSS,
                }, chckpnt_path)
            print('Checkpoint has been saved')
        else:
            print('Current epoch has shown worse result')
            print('Checkpoint has not been saved')
        print('Best result:',best_accuracy)

    #     wandb.log({"train_accuracy":train_accuracy*100,
    #               "val_accuracy":val_accuracy*100,
    #               "loss":train_loss_epoch,
    #               "best_res_epoch":best_res_epoch
    #               })
    
    # wandb.finish()
            


def get_accuracy(model, dataloader, device):
    correct_predicted = 0
    with torch.no_grad():
        for x, y in dataloader:
            # move data to the GPU
            x,y = x.to(0),y.to(0)
            # top 1 predictions
            predictions = torch.argmax(predict(model,x), dim=1, keepdim=True)
            # find the number of correct predictions
            correct_predicted += torch.sum(predictions.eq(y.view_as(predictions))).item()
            # calculate accuracy
            accuracy = correct_predicted / len(dataloader.dataset)
    return accuracy

def load_weights(model, checkpoint_path):
    """
    Initialize `model`'s weights from `checkpoint_path` file.
    
    model:
        `torch.nn.Module`
        See `get_model()`.
    checkpoint_path:
        `str`
        Path to the checkpoint.
    """
    # Your code here
    # get current device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # get model's optimizer
    optimizer = get_optimizer(model)
    # get checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print("Checkpoint has been loaded")
    print("Optimizer has been loaded")
    print("Best params epoch:",epoch)
    print("Loss at this epoch:",loss)

    model.to(device)
    model.eval()
    return model

def get_checkpoint_metadata():
    """
    Return hard-coded metadata for 'checkpoint.pth'.
    Very important for grading.

    return:
    md5_checksum:
        `str`
        MD5 checksum for the submitted 'checkpoint.pth'.
        On Linux (in Colab too), use `$ md5sum checkpoint.pth`.
        On Windows, use `> CertUtil -hashfile checkpoint.pth MD5`.
        On Mac, use `$ brew install md5sha1sum`.
    google_drive_link:
        `str`
        View-only Google Drive link to the submitted 'checkpoint.pth'.
        The file must have the same checksum as in `md5_checksum`.
    """
    # Your code here; md5_checksum = "747822ca4436819145de8f9e410ca9ca"
    # Your code here; google_drive_link = "https://drive.google.com/file/d/1uEwFPS6Gb-BBKbJIfv3hvdaXZ0sdXtOo/view?usp=sharing"
    md5_checksum = "ef823c1a3970c69da9b82beec87512ce"
    google_drive_link = "https://drive.google.com/file/d/1-COfpXvxQLRGb9rcBqBBn40zKlSxOwSs/view?usp=sharing"
    return md5_checksum, google_drive_link
