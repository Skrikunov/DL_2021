import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torch.nn import Conv2d, BatchNorm2d, ReLU, Dropout, MaxPool2d, ConvTranspose2d



class DownBlock(torch.nn.Module):
    def __init__(self, in_chan=None, out_chan=None):
        super(DownBlock, self).__init__()

        self.conv_1_2 = nn.Sequential(
            Conv2d(in_channels=in_chan , out_channels=out_chan, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_chan),
            ReLU(), 
            Conv2d(in_channels=out_chan , out_channels=out_chan , kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_chan),
            ReLU())
        self.mp  = MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        bypass_out = self.conv_1_2(x)
        main_out = self.mp(bypass_out)
        return main_out, bypass_out


class BottleNeckBlock(torch.nn.Module):
    def __init__(self, in_chan=None):
        super(BottleNeckBlock, self).__init__()

        self.conv_1_2 = torch.nn.Sequential(
            Conv2d(in_channels=in_chan, out_channels=in_chan, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(in_chan),
            ReLU(),
            Conv2d(in_channels=in_chan, out_channels=in_chan, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(in_chan),
            ReLU())

    def forward(self, x):
        out = self.conv_1_2(x)
        return out


class UpBlock(torch.nn.Module):
    def __init__(self,in_chan=None, out_chan=None):
        super(UpBlock, self).__init__()

        self.upconv = ConvTranspose2d(in_channels=in_chan, out_channels=in_chan, kernel_size=3, padding=1, output_padding=1 , stride=2)
        self.conv_1_2 = nn.Sequential(
            Dropout(p = 0.5),
            Conv2d(in_channels=2*in_chan, out_channels=out_chan, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_chan),
            ReLU(),
            Conv2d(in_channels = out_chan, out_channels=out_chan, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_chan),
            ReLU())
            
    def forward(self, x_main, x_bypass):
        x_main = self.upconv(x_main) 
        x_main = self.conv_1_2(torch.cat((x_bypass, x_main), dim = 1))
        return x_main 


class UNet(nn.Module):
    """
    TODO: 8 points

    A standard UNet network (with padding in covs).

    For reference, see the scheme in materials/unet.png
    - Use batch norm between conv and relu
    - Use max pooling for downsampling
    - Use conv transpose with kernel size = 3, stride = 2, padding = 1, and output padding = 1 for upsampling
    - Use 0.5 dropout after concat

    Args:
      - num_classes: number of output classes
      - min_channels: minimum number of channels in conv layers
      - max_channels: number of channels in the bottleneck block
      - num_down_blocks: number of blocks which end with downsampling

    The full architecture includes downsampling blocks, a bottleneck block and upsampling blocks

    You also need to account for inputs which size does not divide 2**num_down_blocks:
    interpolate them before feeding into the blocks to the nearest size which divides 2**num_down_blocks,
    and interpolate output logits back to the original shape
    """
    def __init__(self, 
                 num_classes,
                 min_channels=32,
                 max_channels=512, 
                 num_down_blocks=4):
        super(UNet, self).__init__()
        self.num_classes = num_classes

        # TODO
        self.num_down_blocks = num_down_blocks
        # channels for each block
        channels = [32,64,128,256,512]

        # input conv layer
        self.in_Conv2d = Conv2d(in_channels=3,out_channels=min_channels, kernel_size=1)

        # create downblocks
        self.DownBlocks = nn.ModuleList([])
        for i in range(num_down_blocks):
            self.DownBlocks.append(DownBlock(in_chan=channels[i],out_chan=channels[i+1]))

        # middle block
        self.bottleneck_block = BottleNeckBlock(max_channels)

        # create upblocks
        self.UpBlocks = nn.ModuleList([])
        channels.reverse()
        for i in range(num_down_blocks):
            self.UpBlocks.append(UpBlock(in_chan=channels[i],out_chan=channels[i+1]))

        # output conv layer
        self.out_Conv2d = Conv2d(in_channels=min_channels , out_channels=self.num_classes, kernel_size=1)

    def forward(self, inputs):
        # logits = None # TODO

        main = inputs

        parts = 2**self.num_down_blocks
        h_in,w_in = inputs.shape[2],inputs.shape[3]
        h_new,w_new = parts*(h_in//parts),parts*(w_in//parts)

        # do we need interpolation?
        interpolation=False
        if h_in == h_new & w_in == w_new:
            interpolation=False
        else:
            interpolation=True
            self.forward_interpolate = nn.Upsample(size=(h_new, w_new), mode='bilinear', align_corners=False)
            self.inverse_interpolate = nn.Upsample(size=(h_in, w_in), mode='bilinear', align_corners=False)

        bypass_outputs = []
        
        # initial interpolation
        if interpolation:
            main = self.forward_interpolate(main)

        # input conv layer
        main = self.in_Conv2d(main)

        # for each downblock
        for db in self.DownBlocks:
            main, bypass_out = db(main)
            bypass_outputs.append(bypass_out)

        # bypass layer
        main = self.bottleneck_block(main)

        # for each upblock
        bypass_outputs.reverse()
        for ub, bypass_out in zip(self.UpBlocks, bypass_outputs):
            main = ub(main,bypass_out)

        # output conv layer
        main = self.out_Conv2d(main)
        
        # final interpolation
        if interpolation:
            main = self.inverse_interpolate(main)

        # result
        logits = main

        assert logits.shape == (inputs.shape[0], self.num_classes, inputs.shape[2], inputs.shape[3]), 'Wrong shape of the logits'
        return logits





class Block_Conv2d(nn.Sequential):
    def __init__(self,in_channels, out_channels, kernel_size):
        super(Block_Conv2d, self).__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())


class Atrous_Conv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size ,dilation=1 ):
        super(Atrous_Conv2d, self).__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=dilation, dilation=dilation , bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())


class Big_Pooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Big_Pooling,self).__init__()
        self.main_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self,inputs):
        self.upsampling = nn.Upsample(size=(inputs.shape[2],inputs.shape[3]), mode='bilinear', align_corners=False)
        inputs = self.main_layer(inputs)
        inputs = self.upsampling(inputs)
        return inputs


class DeepLab(nn.Module):
    """
    TODO: 6 points

    (simplified) DeepLab segmentation network.
    
    Args:
      - backbone: ['resnet18', 'vgg11_bn', 'mobilenet_v3_small'],
      - aspp: use aspp module
      - num classes: num output classes

    During forward pass:
      - Pass inputs through the backbone to obtain features
      - Apply ASPP (if needed)
      - Apply head
      - Upsample logits back to the shape of the inputs
    """
    def __init__(self, backbone, aspp, num_classes):
        super(DeepLab, self).__init__()
        self.backbone = backbone
        self.init_backbone()

        # ASPP state flag
        self.ASPP_ON = False
        if aspp:
            self.ASPP_ON = True
            self.aspp = ASPP(self.out_features, 256, [12, 24, 36])
        self.num_classes = num_classes

        self.head = DeepLabHead(self.out_features, num_classes)

    def init_backbone(self):
        # TODO: initialize an ImageNet-pretrained backbone
        if self.backbone == 'resnet18':
            self.bb = models.resnet18(pretrained=True)
            self.bb = nn.Sequential(*list(self.bb.children())[:-2])
            self.out_features = 512 # TODO: number of output features in the backbone

        elif self.backbone == 'vgg11_bn':
            self.bb = models.vgg11_bn(pretrained=True)
            self.bb = nn.Sequential(*list(self.bb.children())[:-1][0])
            self.out_features = 512# None # TODO

        elif self.backbone == 'mobilenet_v3_small':
            self.bb = models.mobilenet_v3_small(pretrained=True)
            self.bb = nn.Sequential(*list(self.bb.children())[:-1][0])
            self.out_features = 576 #None # TODO

    def _forward(self, x):
        # TODO: forward pass through the backbone
        if self.backbone == 'resnet18':
            pass

        elif self.backbone == 'vgg11_bn':
            pass

        elif self.backbone == 'mobilenet_v3_small':
            pass

        return x

    def forward(self, inputs):
        # pass # TODO

        # if ASPP on, then use it
        if self.ASPP_ON:
            logits = self.head(self.aspp(self.bb(inputs)))
        # else don't use
        else:
            logits = self.head(self.bb(inputs))
        logits = nn.functional.interpolate(logits, size=(inputs.shape[2],inputs.shape[3]), mode='bilinear', align_corners=False)

        assert logits.shape == (inputs.shape[0], self.num_classes, inputs.shape[2], inputs.shape[3]), 'Wrong shape of the logits'
        return logits


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_classes, 1)
        )


class ASPP(nn.Module):
    """
    TODO: 8 points

    Atrous Spatial Pyramid Pooling module
    with given atrous_rates and out_channels for each head
    Description: https://paperswithcode.com/method/aspp
    
    Detailed scheme: materials/deeplabv3.png
      - "Rates" are defined by atrous_rates
      - "Conv" denotes a Conv-BN-ReLU block
      - "Image pooling" denotes a global average pooling, followed by a 1x1 "conv" block and bilinear upsampling
      - The last layer of ASPP block should be Dropout with p = 0.5

    Args:
      - in_channels: number of input and output channels
      - num_channels: number of output channels in each intermediate "conv" block
      - atrous_rates: a list with dilation values
    """
    def __init__(self, in_channels, num_channels, atrous_rates):
        super(ASPP, self).__init__()

        # pass
        self.blocks = nn.ModuleList([Block_Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=1)])

        for i in range(len(atrous_rates)):
            self.blocks.append(Atrous_Conv2d(in_channels=in_channels, out_channels=num_channels,kernel_size=3,dilation=atrous_rates[i]))
        
        self.blocks.append(Big_Pooling(in_channels=in_channels, out_channels=num_channels))
        self.out_conv2d = Block_Conv2d(in_channels=len(self.blocks)*num_channels,out_channels=in_channels,kernel_size=1)

    def forward(self, x):

        # TODO: forward pass through the ASPP module
        res = []
        for i in range(len(self.blocks)):
            res.append(self.blocks[i](x))
        res = self.out_conv2d(torch.cat(res,dim=1))

        assert res.shape[1] == x.shape[1], 'Wrong number of output channels'
        assert res.shape[2] == x.shape[2] and res.shape[3] == x.shape[3], 'Wrong spatial size'
        return res