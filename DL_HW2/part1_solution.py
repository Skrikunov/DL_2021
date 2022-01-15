# Don't erase the template code, except "Your code here" comments.

import torch
import math                      # Pi

""" Task 1 """

def get_rho():
    # (1) Your code here; theta = ...
    # get pi as doubled arcsin of 1
    pi = 2*torch.asin(torch.tensor(1.))
    theta = torch.linspace(-pi,pi,1000)
    assert theta.shape == (1000,)

    # (2) Your code here; rho = ...
    rho = (1 + 0.9*torch.cos(8*theta))*(1 + 0.1*torch.cos(24*theta))*(0.9 + 0.05*torch.cos(200*theta))*(1 + torch.sin(theta))
    assert torch.is_same_size(rho, theta)
    
    # (3) Your code here; x = ...
    x = rho * torch.cos(theta)
    # (3) Your code here; y = ...
    y = rho * torch.sin(theta)
    return x, y

""" Task 2 """

def game_of_life_update_torch(alive_map):
    """
    PyTorch version of `game_of_life_update_reference()`.
    
    alive_map:
        `torch.tensor`, ndim == 2, dtype == `torch.int64`
        The game map containing 0s (dead) an 1s (alive).
    """
    # Your code here

    # replace the cycle with kernel
    # 2 additional brackets - 2 additional dimentions
    kernel = torch.tensor([[[[1, 1, 1],[1, 0, 1],[1, 1, 1]]]])
    # unsqueeze - to add 2 additional dimentions
    alive_map = torch.unsqueeze(torch.unsqueeze(alive_map,dim = 0),dim = 0)
    # make convolution (calculate neighbours)
    num_alive_neighbors = torch.conv2d(alive_map, kernel, padding=1)

    # Apply game rules

    # do the same as in the loop, bot simultaneously for each cell
    born = (num_alive_neighbors == 3) & (alive_map == 0)
    survived = ((num_alive_neighbors == 2) | (num_alive_neighbors == 3)) & (alive_map == 1)
    new_alive_map = born | survived
    
    # Output the result
    alive_map.copy_(new_alive_map)

""" Task 3 """

# This is a reference layout for encapsulating your neural network. You can add arguments and
# methods if you need to. For example, you may want to add a method `do_gradient_step()` that
# executes one step of an optimization algorithm (SGD / Adadelta / Adam / ...); or you can
# add an extra argument for which you'll find a good value during experiments and set it as
# default to preserve interface (e.g. `def __init__(self, num_hidden_neurons=100):`).
class NeuralNet:
    def __init__(self):

        # Your code here
        in_size = 28*28 # input size of the image
        hid_size = 25 # hidden size

        self.hid_size = hid_size
        # the 1st layer
        self.W1 = torch.randn(in_size, hid_size, requires_grad=True)
        self.b1 = torch.randn(hid_size, requires_grad=True)
        # the 2nd layer
        self.W2 = torch.randn(hid_size, 10, requires_grad=True)
        self.b2 = torch.randn(10, requires_grad=True)
        # gradients
        self.W1_grad_prev = torch.zeros(self.W1.shape)
        self.b1_grad_prev = torch.zeros(self.b1.shape)
        self.W2_grad_prev = torch.zeros(self.W2.shape)
        self.b2_grad_prev = torch.zeros(self.b2.shape)

    def predict(self, images):
        """
        images:
            `torch.tensor`, shape == `batch_size x height x width`, dtype == `torch.float32`
            A minibatch of images -- the input to the neural net.
        
        return:
        prediction:
            `torch.tensor`, shape == `batch_size x 10`, dtype == `torch.float32`
            The scores of each input image to belong to each of the dataset classes.
            Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
            belong to `j`-th class.
            These scores can be 0..1 probabilities, but for better numerical stability
            they can also be raw class scores after the last (usually linear) layer,
            i.e. BEFORE softmax.
        """
        # Your code here
        batches, h, w = images.size()
        images = images.view(batches, h * w)

        self.out = torch.mm(images, self.W1) + self.b1
        self.out = torch.relu(self.out)
        self.out = torch.mm(self.out, self.W2) + self.b2

        return torch.log_softmax(self.out, dim=1)

    # Your code here
    def grad_step_and_zero(self, momentum=0.9, lr=1e-1):
        with torch.no_grad():
            # calculate gradient
            self.W1_grad_prev = momentum*self.W1_grad_prev + lr*self.W1.grad
            self.b1_grad_prev = momentum*self.b1_grad_prev + lr*self.b1.grad
            self.W2_grad_prev = momentum*self.W2_grad_prev + lr*self.W2.grad
            self.b2_grad_prev = momentum*self.b2_grad_prev + lr*self.b2.grad
            # change weights and biases
            self.W1 -= self.W1_grad_prev
            self.b1 -= self.b1_grad_prev
            self.W2 -= self.W2_grad_prev
            self.b2 -= self.b2_grad_prev
            # zero gradient
            self.W1.grad.zero_()
            self.b1.grad.zero_()
            self.W2.grad.zero_()
            self.b2.grad.zero_()

def accuracy(model, images, labels):
    """
    Use `NeuralNet.predict` here.
    
    model:
        `NeuralNet`
    images:
        `torch.tensor`, shape == `N x height x width`, dtype == `torch.float32`
    labels:
        `torch.tensor`, shape == `N`, dtype == `torch.int64`
        Class indices for each sample in `images`.
    
    return:
    value:
        `float`
        The fraction of samples from `images` correctly classified by `model`.
        `0 <= value <= 1`.
    """
    # Your code here
    idx = torch.argmax(model.predict(images), dim=1)
    accuracy_ = torch.mean((idx == labels).type(torch.float32))
    return accuracy_

def train_on_notmnist(model, X_train, y_train, X_val, y_val):
    """
    Update `model`'s weights so that its accuracy on `X_val` is >=82%.
    `X_val`, `y_val` are provided for convenience and aren't required to be used.
    
    model:
        `NeuralNet`
    X_train:
        `torch.tensor`, shape == `N x height x width`, dtype == `torch.float32`
    y_train:
        `torch.tensor`, shape == `N`, dtype == `torch.int64`
        Class indices for each sample in `X_train`.
    X_val, y_val:
        Same as above, possibly with a different length.
    """
    # Your code here
    from IPython import display
    import matplotlib.pyplot as plt

    n_epochs = 100
    # one-hot encoding
    y_train_hot = torch.eye(10)[y_train]
    loss_history = []
    acc_history = []
    batch_size = 512

    for i in range(n_epochs):
        for x_batch, y_batch in get_batches((X_train, y_train_hot), batch_size):

            # forward
            predictions = model.predict(x_batch)
            # get loss
            loss = NLLCriterion(predictions, y_batch)
            # back propagation
            loss.backward()
            # make gradient step and zero gradient
            model.grad_step_and_zero()
            # save current loss
            loss_history.append(loss)
            acc_history.append(accuracy(model, X_val, y_val))
        
        # Visualize
        display.clear_output(wait=True)

        plt.figure(figsize=(8, 6))
        plt.title("Training loss, log scale",fontsize=16)
        plt.xlabel("#iteration",fontsize=16)
        plt.ylabel("loss",fontsize=16)
        plt.yscale('log')
        plt.plot(loss_history, 'b')
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.title("Training accuracy, linear scale",fontsize=16)
        plt.xlabel("#iteration",fontsize=16)
        plt.ylabel("accuracy",fontsize=16)
        plt.plot(acc_history, 'b')
        plt.show()

        accuracy_ = accuracy(model, X_val, y_val)
        print('Current loss: ',loss)
        print('Current accuracy: ',accuracy_)
        # stop if accuracy is higher than required
        if accuracy_ > 0.82:
            break

# batch generator
# this code is from task1, but adopted using pytorch
def get_batches(dataset, batch_size):
    X, Y = dataset
    n_samples = X.shape[0]  
    # Shuffle at the start of epoch
    indices = torch.randperm(n_samples)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        yield X[batch_idx], Y[batch_idx]

# negative log likelyhood criterion
def NLLCriterion(input, target):
    return -torch.sum(torch.sum(target * input, dim=1), dim=0) / input.shape[0]