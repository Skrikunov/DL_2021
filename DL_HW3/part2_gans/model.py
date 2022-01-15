import torch
from torch import nn
from torch.nn import functional as F
import functools
import math
from torch.nn.utils import spectral_norm

class AdaptiveBatchNorm(nn.BatchNorm2d):
    """
    Adaptive batch normalization layer (4 points)

    Args:
        num_features: number of features in batch normalization layer
        embed_features: number of features in embeddings

    The base layer (BatchNorm2d) is applied to "inputs" with affine = False

    After that, the "embeds" are linearly mapped to "gamma" and "bias"
    
    These "gamma" and "bias" are applied to the outputs like in batch normalization
    with affine = True (see definition of batch normalization for reference)
    """
    def __init__(self, num_features: int, embed_features: int):
        super(AdaptiveBatchNorm, self).__init__(num_features, affine=False)
        
        self.num_features = num_features
        self.embed_features = embed_features
        self.lay1 = spectral_norm(nn.Linear(in_features=self.embed_features,out_features=self.num_features))
        self.lay2 = spectral_norm(nn.Linear(in_features=self.embed_features,out_features=self.num_features))

    def forward(self, inputs, embeds):
        gamma = self.lay1(embeds) # TODO 
        bias = self.lay2(embeds) # TODO

        assert gamma.shape[0] == inputs.shape[0] and gamma.shape[1] == inputs.shape[1]
        assert bias.shape[0] == inputs.shape[0] and bias.shape[1] == inputs.shape[1]

        outputs = super().forward(inputs) # TODO: apply batchnorm

        return outputs * gamma[..., None, None] + bias[..., None, None]


class Pre_Act_Block_Adapt_Batch_Norm(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, embed_features: int) -> None:
        super(Pre_Act_Block_Adapt_Batch_Norm, self).__init__(
            AdaptiveBatchNorm(in_channels,embed_features),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,bias=False,padding=1)))
    def forward(self, inputs,embeds):
        for idx, func in enumerate(self):
            if not idx:
                inputs = func(inputs,embeds)
            else:
                inputs = func(inputs)
        return inputs


class Pre_Act_Block(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Pre_Act_Block, self).__init__(
            nn.ReLU(),
            spectral_norm(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,bias=False,padding=1)))


class PreActResBlock(nn.Module):
    """
    Pre-activation residual block (6 points)

    Paper: https://arxiv.org/pdf/1603.05027.pdf
    Scheme: materials/preactresblock.png
    Review: https://towardsdatascience.com/resnet-with-identity-mapping-over-1000-layers-reached-image-classification-bb50a42af03e

    Args:
        in_channels: input number of channels
        out_channels: output number of channels
        batchnorm: this block is with/without adaptive batch normalization
        upsample: use nearest neighbours upsampling at the beginning
        downsample: use average pooling after the end

    in_channels != out_channels:
        - first conv: in_channels -> out_channels
        - second conv: out_channels -> out_channels
        - use 1x1 conv in skip connection

    in_channels == out_channels: skip connection is without a conv
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 embed_channels: int = None,
                 batchnorm: bool = False,
                 upsample: bool = False,
                 downsample: bool = False):
        super(PreActResBlock, self).__init__()

        # TODO: define pre-activation residual block
        # TODO: apply spectral normalization to conv layers
        # Don't forget that activation after residual sum cannot be inplace!
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_channels = embed_channels
        self.batchnorm = batchnorm
        self.upsample = upsample
        self.downsample = downsample

        if self.batchnorm:
            pa1 = Pre_Act_Block_Adapt_Batch_Norm(in_channels=self.in_channels,out_channels=self.out_channels,embed_features=self.embed_channels)
            pa2 = Pre_Act_Block_Adapt_Batch_Norm(in_channels=self.out_channels,out_channels=self.out_channels,embed_features=self.embed_channels)
        else:
            pa1 = Pre_Act_Block(in_channels=self.in_channels,out_channels=self.out_channels)
            pa2 = Pre_Act_Block(in_channels=self.out_channels,out_channels=self.out_channels)
        self.ModuleList = nn.ModuleList([pa1, pa2])

        self.bypass = spectral_norm(nn.Conv2d(in_channels=self.in_channels,out_channels=self.out_channels,kernel_size=1))

        if self.downsample:
            self.ap = nn.AvgPool2d(2)

    def forward(self, 
                inputs, # regular features 
                embeds=None): # embeds used in adaptive batch norm

        # TODO
        if self.upsample:
            inputs = F.interpolate(inputs, scale_factor=2)

        x = inputs
        if self.batchnorm:
            for i in range(len(self.ModuleList)):
                inputs = self.ModuleList[i](inputs, embeds) 
        else:
            for i in range(len(self.ModuleList)):
                inputs = self.ModuleList[i](inputs) 

        if self.in_channels==self.out_channels:
            inputs = inputs + x
        else:
            inputs = inputs + self.bypass(x)

        if self.downsample:
            inputs = self.ap(inputs)
        outputs = inputs

        return outputs



class Generator(nn.Module):
    """
    Generator network (8 points)
    
    TODO:

      - Implement an option to condition the synthesis on trainable class embeddings
        (use nn.Embedding module with noise_channels as the size of each embed)

      - Concatenate input noise with class embeddings (if use_class_condition = True) to obtain input embeddings

      - Linearly map input embeddings into input tensor with the following dims: max_channels x 4 x 4

      - Forward an input tensor through a convolutional part, 
        which consists of num_blocks PreActResBlocks and performs upsampling by a factor of 2 in each block

      - Each PreActResBlock is additionally conditioned on the input embeddings (via adaptive batch normalization)

      - At the end of the convolutional part apply regular BN, ReLU and Conv as an image prediction head

      - Apply spectral norm to all conv and linear layers (not the embedding layer)

      - Use Sigmoid at the end to map the outputs into an image

    Notes:

      - The last convolutional layer should map min_channels to 3. With each upsampling you should decrease
        the number of channels by a factor of 2

      - Class embeddings are only used and trained if use_class_condition = True
    """    
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 noise_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 use_class_condition: bool):
        super(Generator, self).__init__()
        self.output_size = 4 * 2**num_blocks

        # TODO
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.noise_channels = noise_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.use_class_condition = use_class_condition

        if self.use_class_condition:
            self.embed_layer = nn.Embedding(num_embeddings=self.num_classes,embedding_dim=self.noise_channels)
            self.lin_layer = spectral_norm(nn.Linear(in_features=2*self.noise_channels,out_features=4*4*self.max_channels))
        else:
            self.lin_layer = spectral_norm(nn.Linear(in_features=self.noise_channels,out_features=4*4*self.max_channels))

        ModuleList = nn.ModuleList()
        in_channels = self.max_channels
        out_channels = int(in_channels/2)
        for _ in range(self.num_blocks):
            if self.use_class_condition:
                ModuleList.append(PreActResBlock(in_channels,out_channels,2*self.noise_channels,batchnorm=True,upsample=True))
            else:
                ModuleList.append(PreActResBlock(in_channels,out_channels,self.noise_channels,batchnorm=False,upsample=True))
            in_channels = out_channels
            out_channels = int(in_channels/2)
        self.Module_List = nn.Sequential(*ModuleList)

        self.final_block = nn.Sequential(
            nn.BatchNorm2d(self.min_channels),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(in_channels=self.min_channels,out_channels=3,kernel_size=3,padding=1)),
            nn.Sigmoid())

    def forward(self, noise, labels):
        # TODO
        # noise.shape B x noise_channels
        # labels.shape B

        if self.use_class_condition:
            embeds = self.embed_layer(labels)
            noise = torch.cat((noise, embeds),dim=1)
            embeds = noise 

        batch_size = noise.shape[0]
        noise = self.lin_layer(noise).reshape(batch_size,self.max_channels,4,4)

        for i in range(len(self.Module_List)):
            if self.use_class_condition:
                noise = self.Module_List[i](noise,embeds)
            else:
                noise = self.Module_List[i](noise)
        outputs = self.final_block(noise)

        assert outputs.shape == (noise.shape[0], 3, self.output_size, self.output_size)
        return outputs



class Discriminator(nn.Module):
    """
    Discriminator network (8 points)

    TODO:
    
      - Define a convolutional part of the discriminator similarly to
        the generator blocks, but in the inverse order, with downsampling, and
        without batch normalization
    
      - At the end of the convolutional part apply ReLU and sum pooling
    
    TODO: implement projection discriminator head (https://arxiv.org/abs/1802.05637)
    
    Scheme: materials/prgan.png
    
    Notation:
    
      - phi is a convolutional part of the discriminator
    
      - psi is a vector
    
      - y is a class embedding
    
    Class embeddings matrix is similar to the generator, shape: num_classes x max_channels

    Discriminator outputs a B x 1 matrix of realism scores

    Apply spectral norm for all layers (conv, linear, embedding)
    """
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 use_projection_head: bool):
        super(Discriminator, self).__init__()
        # TODO
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.use_projection_head = use_projection_head

        self.init_layer = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(in_channels=3, out_channels=self.min_channels,kernel_size=3,padding=1)))

        ModuleList = nn.ModuleList()
        in_channels = self.min_channels
        out_channels = 2*in_channels
        for _ in range(self.num_blocks):
            ModuleList.append(PreActResBlock(in_channels,out_channels,self.max_channels,batchnorm=False,downsample=True))
            in_channels = out_channels
            out_channels = 2*in_channels
        self.Module_List = nn.Sequential(*ModuleList)

        self.ReLU = nn.ReLU()

        self.lin_layer = spectral_norm(nn.Linear(in_features=self.max_channels,out_features=1))
        self.embed_layer = spectral_norm(nn.Embedding(num_embeddings=self.num_classes,embedding_dim=self.max_channels))

    def forward(self, inputs, labels):
        inputs = self.init_layer(inputs)

        for i in range(len(self.Module_List)):
            inputs = self.Module_List[i](inputs)

        inputs = self.ReLU(inputs)
        inputs = inputs.sum(dim=[2,3])
        scores = self.lin_layer(inputs).flatten()
        scores = scores.flatten()

        if self.use_projection_head:
            in_prod = self.embed_layer(labels)*inputs
            scores = scores + in_prod.sum(dim=1)

        assert scores.shape == (inputs.shape[0],)
        return scores