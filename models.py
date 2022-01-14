from torch.nn.modules import dropout
from dataloader import PictureLoader, image_saver, epoch_image_saver
from torchsummary import summary
import torchvision.models.vgg as vgg
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import datetime
import time
from gradflowchecker import CheckGradFlow
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dropout=True):
        super(Upsample, self).__init__()
        self.dropout = dropout
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                               stride, padding, bias=nn.InstanceNorm2d),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
        self.dropout_layer = nn.Dropout2d(0.5)

        nn.init.kaiming_uniform_(self.block[0].weight)

    def forward(self, x, shortcut=None):
        x = self.block(x)
        if self.dropout:
            x = self.dropout_layer(x)
        if shortcut is not None:
            x = torch.cat([x, shortcut], dim=1)

        return x


class UnetGenerator(nn.Module):
    def __init__(self, filter=64, in_channels=3, negative_slope=0.2, dropout_rate=0, inplace=True):
        super(UnetGenerator, self).__init__()

        self.downsample1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filter,
                      kernel_size=4, stride=2, padding=1, bias=nn.InstanceNorm2d),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(in_channels=filter, out_channels=filter*2,
                      kernel_size=4, stride=2, padding=1, bias=nn.InstanceNorm2d),
            nn.InstanceNorm2d(num_features=filter*2),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(in_channels=filter*2, out_channels=filter*4,
                      kernel_size=4, stride=2, padding=1, bias=nn.InstanceNorm2d),
            nn.InstanceNorm2d(num_features=filter*4),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
        )
        self.downsample4 = nn.Sequential(
            nn.Conv2d(in_channels=filter*4, out_channels=filter*8,
                      kernel_size=4, stride=2, padding=1, bias=nn.InstanceNorm2d),
            nn.InstanceNorm2d(num_features=filter*8),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
        )

        self.downsample5 = nn.Sequential(
            nn.Conv2d(in_channels=filter*8, out_channels=filter*8,
                      kernel_size=4, stride=2, padding=1, bias=nn.InstanceNorm2d),
            nn.InstanceNorm2d(num_features=filter*8),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
        )
        #(256, 8, 8)

        self.upsample1 = Upsample(filter*8, filter*8)
        self.upsample2 = Upsample(filter*16, filter*4, dropout=False)
        self.upsample3 = Upsample(filter*8, filter*2, dropout=False)
        self.upsample4 = Upsample(filter*4, filter, dropout=False)

        self.last_conv = nn.Sequential(
            nn.ConvTranspose2d(filter*2, 3, kernel_size=4,
                               stride=2, padding=1),
            nn.Tanh()
        )

        nn.init.kaiming_uniform_(self.downsample1[0].weight)
        nn.init.kaiming_uniform_(self.downsample2[0].weight)
        nn.init.kaiming_uniform_(self.downsample3[0].weight)
        nn.init.kaiming_uniform_(self.downsample4[0].weight)
        nn.init.kaiming_uniform_(self.downsample5[0].weight)
        nn.init.xavier_uniform_(self.last_conv[0].weight)

    def forward(self, img):
        d1 = self.downsample1(img)  # (64, 128, 128)
        d2 = self.downsample2(d1)  # (128, 64, 64)
        d3 = self.downsample3(d2)  # (256, 32, 32)
        d4 = self.downsample4(d3)  # (512, 16, 16)
        d5 = self.downsample5(d4)  # (512, 8, 8)

        u1 = self.upsample1(d5, d4)  # (1024, 16, 16)
        u2 = self.upsample2(u1, d3)  # (512, 32, 32)
        u3 = self.upsample3(u2, d2)  # (256, 64, 64)
        u4 = self.upsample4(u3, d1)  # (128, 128, 128)

        output = self.last_conv(u4)

        return output


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout_rate=0, negative_slope=0.2, leaky=True, inplace=True, b_norm=True, apool=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.b_norm_2d = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ReLU(inplace=inplace)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.ave_pooling = nn.AvgPool2d(kernel_size=4, stride=2, padding=1)
        self.b_norm = b_norm
        self.leaky = leaky
        self.apool = apool
        if leaky:
            nn.init.kaiming_normal_(self.conv.weight, a=0.2)
        else:
            nn.init.xavier_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.b_norm:
            x = self.b_norm_2d(x)
        if self.leaky:
            x = self.activation(x)
        x = self.dropout(x)
        if self.apool:
            x = self.ave_pooling(x)

        return x


class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, negative_slope=0.2, inplace=True, ref_pad=False, instance_norm=True, acti=True):
        super(DeconvLayer, self).__init__()

        self.refrection_pad = nn.ReplicationPad2d(padding=padding)
        if ref_pad:
            padding = 0
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.insta_norm = nn.InstanceNorm2d(num_features=out_channels)
        self.leaky_activation = nn.LeakyReLU(
            negative_slope=negative_slope, inplace=True)
        self.ref_pad = ref_pad
        self.instance_norm = instance_norm
        self.acti = acti

        if acti:
            nn.init.kaiming_uniform_(self.deconv.weight)
        else:
            nn.init.xavier_uniform_(self.deconv.weight)
        nn.init.constant_(self.deconv.bias, 0)

    def forward(self, x):
        if self.ref_pad:
            x = self.refrection_pad(x)
        x = self.deconv(x)
        if self.instance_norm:
            x = self.insta_norm(x)
        if self.acti:
            x = self.leaky_activation(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, negative_slope=0.2, dropout_rate_input_layer=0, dropout_rate_embed_layer=0.3, inplace=True):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace),
            nn.Dropout2d(p=dropout_rate_input_layer)
        )
        # (64, 128, 128)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace),
            nn.Dropout2d(p=dropout_rate_embed_layer)
        )
        # (128, 64, 64)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace),
            nn.Dropout2d(p=dropout_rate_embed_layer)
        )
        # (256, 32, 32)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256, affine=False),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace),
            nn.Dropout2d(p=dropout_rate_embed_layer)
        )
        # (256, 31, 31)

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1,
                      kernel_size=4, stride=1, padding=1)
        )
        # (1, 30, 30)

        nn.init.kaiming_normal_(self.conv1[0].weight)
        nn.init.kaiming_normal_(self.conv2[0].weight)
        nn.init.kaiming_normal_(self.conv3[0].weight)
        nn.init.kaiming_normal_(self.conv4[0].weight)
        nn.init.xavier_normal_(self.conv5[0].weight)

    def forward(self, img):
        y = self.conv1(img)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
        return y


class Discriminator_1(nn.Module):
    def __init__(self):
        super(Discriminator_1, self).__init__()

        # input=(3, 256, 256)

        self.conv1 = ConvLayer(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        #(64, 256, 256)

        self.conv2 = ConvLayer(in_channels=64, out_channels=128, kernel_size=4,
                               stride=2, padding=1, dropout_rate=0.3, apool=False)
        #(128, 128, 128)

        self.conv3 = ConvLayer(in_channels=128, out_channels=256,
                               kernel_size=3, stride=1, padding=1, dropout_rate=0.3)
        #(256, 128, 128)

        self.conv4 = ConvLayer(in_channels=256, out_channels=512, kernel_size=4,
                               stride=2, padding=1, dropout_rate=0.3, apool=False)
        #(512, 64, 64)

        self.conv5 = ConvLayer(in_channels=512, out_channels=512,
                               kernel_size=3, stride=1, padding=1, dropout_rate=0.3)
        #(512, 64, 64)

        self.conv6 = ConvLayer(in_channels=512, out_channels=1, kernel_size=3,
                               stride=1, padding=1, dropout_rate=0.3, apool=False)
        #(1, 64, 64)

    def forward(self, img):
        y = self.conv1(img)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
        y = self.conv6(y)

        return y


class ResidualBrock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(ResidualBrock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.activation1 = nn.LeakyReLU(inplace=True)
        self.activation2 = nn.LeakyReLU(inplace=True)
        self.batchNorm1 = nn.InstanceNorm2d(
            num_features=mid_channels, affine=False)
        self.batchNorm2 = nn.InstanceNorm2d(
            num_features=out_channels, affine=False)
        self.dropout1 = nn.Dropout2d()
        self.dropout2 = nn.Dropout2d()
        
        #nn.init.kaiming_uniform_(self.conv1.weight)
        #nn.init.kaiming_uniform_(self.conv2.weight)

    def forward(self, x):
        residual = x

        conved = self.conv1(x)
        conved = self.batchNorm1(conved)
        conved = self.activation1(conved)
        conved = self.dropout1(conved)
        
        conved = self.conv2(conved)
        conved = self.batchNorm2(conved)

        output = conved + residual

        output = self.activation2(output)
        return output


class ResnetGenerator9(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, gen_filters=64, inplace=True):
        super(ResnetGenerator9, self).__init__()

        self.pre_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=in_channels, out_channels=gen_filters, kernel_size=7,
                      stride=1, padding=0, bias=nn.InstanceNorm2d),  # (64, 256, 256)
            nn.BatchNorm2d(num_features=gen_filters),
            nn.ReLU(inplace=inplace)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=gen_filters, out_channels=gen_filters*2, kernel_size=4,
                      stride=2, padding=1, bias=nn.InstanceNorm2d),  # (128, 128, 128)
            nn.BatchNorm2d(num_features=gen_filters*2),
            nn.ReLU(inplace=inplace)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=gen_filters*2, out_channels=gen_filters*4, kernel_size=4,
                      stride=2, padding=1, bias=nn.InstanceNorm2d),  # (256, 64, 64)
            nn.BatchNorm2d(num_features=gen_filters*4),
            nn.ReLU(inplace=inplace)
        )

        self.residual1 = ResidualBrock(
            in_channels=gen_filters*4, mid_channels=gen_filters*4, out_channels=gen_filters*4)
        self.residual2 = ResidualBrock(
            in_channels=gen_filters*4, mid_channels=gen_filters*4, out_channels=gen_filters*4)
        self.residual3 = ResidualBrock(
            in_channels=gen_filters*4, mid_channels=gen_filters*4, out_channels=gen_filters*4)
        self.residual4 = ResidualBrock(
            in_channels=gen_filters*4, mid_channels=gen_filters*4, out_channels=gen_filters*4)
        self.residual5 = ResidualBrock(
            in_channels=gen_filters*4, mid_channels=gen_filters*4, out_channels=gen_filters*4)
        self.residual6 = ResidualBrock(
            in_channels=gen_filters*4, mid_channels=gen_filters*4, out_channels=gen_filters*4)
        self.residual7 = ResidualBrock(
            in_channels=gen_filters*4, mid_channels=gen_filters*4, out_channels=gen_filters*4)
        self.residual8 = ResidualBrock(
            in_channels=gen_filters*4, mid_channels=gen_filters*4, out_channels=gen_filters*4)
        self.residual9 = ResidualBrock(
            in_channels=gen_filters*4, mid_channels=gen_filters*4, out_channels=gen_filters*4)  # (256, 64, 64)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=gen_filters*4, out_channels=gen_filters*2,
                               kernel_size=4, stride=2, padding=1, bias=nn.InstanceNorm2d),  # (128, 128, 128)
            nn.BatchNorm2d(num_features=gen_filters*2),
            nn.ReLU(inplace=inplace)
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=gen_filters*2, out_channels=gen_filters,
                               kernel_size=4, stride=2, padding=1, bias=nn.InstanceNorm2d),  # (64, 256, 256)
            nn.BatchNorm2d(num_features=gen_filters),
            nn.ReLU(inplace=inplace)
        )

        self.last_deconv = nn.Sequential(
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(in_channels=gen_filters, out_channels=out_channels,
                      kernel_size=7, stride=1, padding=0),  # (3, 256, 256)
            nn.Tanh()
        )

        nn.init.kaiming_normal_(self.pre_conv[1].weight)
        nn.init.kaiming_normal_(self.conv1[0].weight)
        nn.init.kaiming_normal_(self.conv2[0].weight)
        nn.init.kaiming_normal_(self.deconv1[0].weight)
        nn.init.kaiming_normal_(self.deconv2[0].weight)
        nn.init.xavier_normal_(self.last_deconv[1].weight)

        nn.init.constant_(self.last_deconv[1].bias, 0)

    def forward(self, x):
        x = self.pre_conv(x)  # (64, 256, 256)
        x = self.conv1(x)  # (128, 128, 128)
        x = self.conv2(x)  # (256, 64, 64)

        x = self.residual1(x)  # (256, 64, 64)
        x = self.residual2(x)  # (256, 64, 64)
        x = self.residual3(x)  # (256, 64, 64)
        x = self.residual4(x)  # (256, 64, 64)
        x = self.residual5(x)  # (256, 64, 64)
        x = self.residual6(x)  # (256, 64, 64)
        x = self.residual7(x)  # (256, 64, 64)
        x = self.residual8(x)  # (256, 64, 64)
        x = self.residual9(x)  # (256, 64, 64)

        x = self.deconv1(x)  # (128, 128, 128)
        x = self.deconv2(x)  # (64, 256, 256)
        x = self.last_deconv(x)  # (3, 256, 256)

        return x


class ResnetGenerator6(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, gen_filters=64, inplace=True):
        super(ResnetGenerator6, self).__init__()

        self.pre_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=in_channels, out_channels=gen_filters,
                      kernel_size=7, stride=1, padding=0),  # (64, 256, 256)
            nn.BatchNorm2d(num_features=gen_filters, affine=False),
            nn.LeakyReLU(inplace=inplace)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=gen_filters, out_channels=gen_filters*2,
                      kernel_size=4, stride=2, padding=1),  # (128, 128, 128)
            nn.BatchNorm2d(num_features=gen_filters*2, affine=False),
            nn.LeakyReLU(inplace=inplace)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=gen_filters*2, out_channels=gen_filters*4,
                      kernel_size=4, stride=2, padding=1),  # (256, 64, 64)
            nn.BatchNorm2d(num_features=gen_filters*4, affine=False),
            nn.LeakyReLU(inplace=inplace)
        )

        self.residual1 = ResidualBrock(
            in_channels=gen_filters*4, mid_channels=gen_filters*4, out_channels=gen_filters*4)
        self.residual2 = ResidualBrock(
            in_channels=gen_filters*4, mid_channels=gen_filters*4, out_channels=gen_filters*4)
        self.residual3 = ResidualBrock(
            in_channels=gen_filters*4, mid_channels=gen_filters*4, out_channels=gen_filters*4)
        self.residual4 = ResidualBrock(
            in_channels=gen_filters*4, mid_channels=gen_filters*4, out_channels=gen_filters*4)
        self.residual5 = ResidualBrock(
            in_channels=gen_filters*4, mid_channels=gen_filters*4, out_channels=gen_filters*4)
        self.residual6 = ResidualBrock(
            in_channels=gen_filters*4, mid_channels=gen_filters*4, out_channels=gen_filters*4)  # (256, 64, 64)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=gen_filters*4, out_channels=gen_filters*2,
                               kernel_size=4, stride=2, padding=1),  # (128, 128, 128)
            nn.InstanceNorm2d(num_features=gen_filters*2, affine=False),
            nn.LeakyReLU(inplace=inplace),
            nn.Dropout2d()
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=gen_filters*2, out_channels=gen_filters,
                               kernel_size=4, stride=2, padding=1),  # (64, 256, 256)
            nn.InstanceNorm2d(num_features=gen_filters, affine=False),
            nn.LeakyReLU(inplace=inplace),
            nn.Dropout2d()
        )

        self.last_deconv = nn.Sequential(
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(in_channels=gen_filters, out_channels=out_channels,
                      kernel_size=7, stride=1, padding=0),  # (3, 256, 256)
            nn.InstanceNorm2d(num_features=out_channels, affine=False),
            nn.Tanh()
        )

        nn.init.kaiming_normal_(self.pre_conv[1].weight)
        nn.init.kaiming_normal_(self.conv1[0].weight)
        nn.init.kaiming_normal_(self.conv2[0].weight)
        nn.init.kaiming_normal_(self.deconv1[0].weight)
        nn.init.kaiming_normal_(self.deconv2[0].weight)
        nn.init.xavier_normal_(self.last_deconv[1].weight)

        nn.init.constant_(self.pre_conv[1].bias, 0)
        nn.init.constant_(self.conv1[0].bias, 0)
        nn.init.constant_(self.conv2[0].bias, 0)
        nn.init.constant_(self.deconv1[0].bias, 0)
        nn.init.constant_(self.deconv1[0].bias, 0)
        nn.init.constant_(self.last_deconv[1].bias, 0)

    def forward(self, x):
        x = self.pre_conv(x)  # (64, 256, 256)
        x = self.conv1(x)  # (128, 128, 128)
        x = self.conv2(x)  # (256, 64, 64)

        x = self.residual1(x)  # (256, 64, 64)
        x = self.residual2(x)  # (256, 64, 64)
        x = self.residual3(x)  # (256, 64, 64)
        x = self.residual4(x)  # (256, 64, 64)
        x = self.residual5(x)  # (256, 64, 64)
        x = self.residual6(x)  # (256, 64, 64)

        x = self.deconv1(x)  # (128, 128, 128)
        x = self.deconv2(x)  # (64, 256, 256)
        x = self.last_deconv(x)  # (3, 256, 256)

        return x


class ResnetGenerator3(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, gen_filters=64, inplace=True):
        super(ResnetGenerator3, self).__init__()

        self.pre_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=in_channels, out_channels=gen_filters, kernel_size=7,
                      stride=1, padding=0, bias=nn.InstanceNorm2d),  # (64, 256, 256)
            nn.BatchNorm2d(num_features=gen_filters),
            nn.ReLU(inplace=inplace)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=gen_filters, out_channels=gen_filters*2, kernel_size=4,
                      stride=2, padding=1, bias=nn.InstanceNorm2d),  # (128, 128, 128)
            nn.BatchNorm2d(num_features=gen_filters*2),
            nn.ReLU(inplace=inplace)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=gen_filters*2, out_channels=gen_filters*4, kernel_size=4,
                      stride=2, padding=1, bias=nn.InstanceNorm2d),  # (256, 64, 64)
            nn.BatchNorm2d(num_features=gen_filters*4),
            nn.ReLU(inplace=inplace)
        )

        self.residual1 = ResidualBrock(
            in_channels=gen_filters*4, mid_channels=gen_filters*4, out_channels=gen_filters*4)
        self.residual2 = ResidualBrock(
            in_channels=gen_filters*4, mid_channels=gen_filters*4, out_channels=gen_filters*4)
        self.residual3 = ResidualBrock(
            in_channels=gen_filters*4, mid_channels=gen_filters*4, out_channels=gen_filters*4)  # (256, 64, 64)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=gen_filters*4, out_channels=gen_filters*2,
                               kernel_size=4, stride=2, padding=1, bias=nn.InstanceNorm2d),  # (128, 128, 128)
            nn.BatchNorm2d(num_features=gen_filters*2),
            nn.ReLU(inplace=inplace)
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=gen_filters*2, out_channels=gen_filters,
                               kernel_size=4, stride=2, padding=1, bias=nn.InstanceNorm2d),  # (64, 256, 256)
            nn.BatchNorm2d(num_features=gen_filters),
            nn.ReLU(inplace=inplace)
        )

        self.last_deconv = nn.Sequential(
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(in_channels=gen_filters, out_channels=out_channels,
                      kernel_size=7, stride=1, padding=0),  # (3, 256, 256)
            nn.Tanh()
        )

        nn.init.kaiming_normal_(self.pre_conv[1].weight)
        nn.init.kaiming_normal_(self.conv1[0].weight)
        nn.init.kaiming_normal_(self.conv2[0].weight)
        nn.init.kaiming_normal_(self.deconv1[0].weight)
        nn.init.kaiming_normal_(self.deconv2[0].weight)
        nn.init.xavier_normal_(self.last_deconv[1].weight)

        nn.init.constant_(self.last_deconv[1].bias, 0)

    def forward(self, x):
        x = self.pre_conv(x)  # (64, 256, 256)
        x = self.conv1(x)  # (128, 128, 128)
        x = self.conv2(x)  # (256, 64, 64)

        x = self.residual1(x)  # (256, 64, 64)
        x = self.residual2(x)  # (256, 64, 64)
        x = self.residual3(x)  # (256, 64, 64)

        x = self.deconv1(x)  # (128, 128, 128)
        x = self.deconv2(x)  # (64, 256, 256)
        x = self.last_deconv(x)  # (3, 256, 256)

        return x


def weight_saver(gen_B, gen_A):
    time_now = datetime.datetime.now()
    time_info = f'{time_now.year}-{time_now.month}-{time_now.day}_{time_now.hour}-{time_now.minute}-{time_now.second}'

    save_path_A2B = './weight/A2B/trained_weight_A2B_'+str(time_info)+'.pth'
    save_path_B2A = './weight/B2A/trained_weight_B2A_'+str(time_info)+'.pth'

    try:
        torch.save(gen_B.state_dict(), save_path_A2B)
        torch.save(gen_A.state_dict(), save_path_B2A)
        print('Trained parameters were successfully saved!')
    except:
        print('Trained parameters were not successfully saved!')

    return None


def log_show(logs, save_bool=True):
    time_now = datetime.datetime.now()
    time_info = f'{time_now.year}-{time_now.month}-{time_now.day}_{time_now.hour}-{time_now.minute}-{time_now.second}'

    save_path = './pictures/logs/'+str(time_info)+'.jpg'

    size = logs[0].shape[0]
    x = [num for num in range(size)]

    dec = logs[0].tolist()
    gen = logs[1].tolist()

    plt.plot(x, dec, color='red', label='discriminator')
    plt.plot(x, gen, color='blue', label='generator')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')

    if save_bool:
        plt.savefig(save_path)

    plt.show()
    return None


def train_model(gen_B, gen_A, dec_A, dec_B, dataloader, num_epochs, cycle_loss_rate=10, identity_loss_rate=5, toReturnLoss=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    torch.backends.cudnn.benchmark = True
    # Difine Loss

    D_A_Loss_1 = nn.MSELoss(reduction='mean')
    D_B_Loss_1 = nn.MSELoss(reduction='mean')

    D_A_Loss_2 = nn.MSELoss(reduction='mean')  # (dec_gan_A, 0)
    D_B_Loss_2 = nn.MSELoss(reduction='mean')  # (dec_gan_B, 0)

    g_Loss_B_1 = nn.MSELoss(reduction='mean')
    g_Loss_A_1 = nn.MSELoss(reduction='mean')

    cycle_Loss_A = nn.L1Loss(reduction='mean')  # input_A - cyc_A
    cycle_Loss_B = nn.L1Loss(reduction='mean')  # input_B - cyc_B

    identity_Loss_A = nn.L1Loss(reduction='mean')
    identity_Loss_B = nn.L1Loss(reduction='mean')

    # Optimizer

    beta1, beta2 = 0.9, 0.99
    lr_g = 0.0008  # 0.0004
    lr_d = 0.0008  # 0.0008

    optimizer_gen_B = optim.Adam(
        params=gen_B.parameters(), lr=lr_g, betas=(beta1, beta2))
    optimizer_gen_A = optim.Adam(
        params=gen_A.parameters(), lr=lr_g, betas=(beta1, beta2))
    optimizer_dec_A = optim.Adam(
        params=dec_A.parameters(), lr=lr_d, betas=(beta1, beta2))
    optimizer_dec_B = optim.Adam(
        params=dec_B.parameters(), lr=lr_d, betas=(beta1, beta2))
    
    optimizer_gen_B = optim.SGD(
        params=gen_B.parameters(), lr=lr_g, weight_decay=0.0001)
    optimizer_gen_A = optim.SGD(
        params=gen_A.parameters(), lr=lr_g, weight_decay=0.0001)
    optimizer_dec_A = optim.SGD(
        params=dec_A.parameters(), lr=lr_d, weight_decay=0.0001)
    optimizer_dec_B = optim.SGD(
        params=dec_B.parameters(), lr=lr_d, weight_decay=0.0001)

    scheduler_gen_B = optim.lr_scheduler.StepLR(
        optimizer=optimizer_gen_B, step_size=15, gamma=0.5)
    scheduler_gen_A = optim.lr_scheduler.StepLR(
        optimizer=optimizer_gen_A, step_size=15, gamma=0.5)
    scheduler_dec_A = optim.lr_scheduler.StepLR(
        optimizer=optimizer_dec_A, step_size=15, gamma=0.5)
    scheduler_dec_B = optim.lr_scheduler.StepLR(
        optimizer=optimizer_dec_B, step_size=15, gamma=0.5)

    gradFlowChecker = CheckGradFlow()

    # Train the model

    batch_size = dataloader.batch_size

    gen_B.to(device)
    gen_A.to(device)
    dec_A.to(device)
    dec_B.to(device)

    logs = np.zeros((2, num_epochs+1), dtype='float32')

    for epoch in range(num_epochs):
        flag = True
        epoch_start_time = time.perf_counter()

        epoch_D_gen_loss = 0.0
        epoch_total_gen_loss = 0.0
        epoch_total_dec_loss = 0.0
        epoch_total_gen_loss_unit = 0.0

        epoch_valid_loss = 0.0
        epoch_cycle_loss = 0.0
        epoch_identity_loss = 0.0

        print(
            '----------------------------------------------------------------------------')
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(
            '----------------------------------------------------------------------------')
        print('(Train)')

        for images in tqdm(dataloader):
            img_A, img_B = images[0], images[1]

            img_A = img_A.to(device)
            img_B = img_B.to(device)

            # ========================================================================
            # Train generators with discriminators' parameters flozen
            # ========================================================================

            # Valid Loss
            generated_A = gen_A(img_B)
            generated_B = gen_B(img_A)

            judged_B = dec_B(generated_B.detach())  # judge = valid
            judged_A = dec_A(generated_A.detach())  # judge = valid

            judged_B_label = torch.ones_like(judged_B).to(device)
            judged_A_label = torch.ones_like(judged_A).to(device)

            g_loss_B_1 = g_Loss_B_1(judged_B, judged_B_label)
            g_loss_A_1 = g_Loss_A_1(judged_A, judged_A_label)
            valid_loss = (g_loss_B_1 + g_loss_A_1) / 2
            epoch_valid_loss += valid_loss.item()

            # Cycle loss
            reconstr_A = gen_A(generated_B)
            reconstr_B = gen_B(generated_A)

            cycle_loss_A = cycle_Loss_A(img_A, reconstr_A)  # input_A - cyc_A
            cycle_loss_B = cycle_Loss_B(img_B, reconstr_B)  # input_B - cyc_B
            cycle_loss = (cycle_loss_A + cycle_loss_B) / 2  # Cycle_Loss
            epoch_cycle_loss += cycle_loss.item()

            # Identity loss
            identity_B = gen_B(img_B)
            identity_A = gen_A(img_A)

            identity_B_loss = identity_Loss_B(identity_B, img_B)
            identity_A_loss = identity_Loss_A(identity_A, img_A)

            identity_loss = (identity_B_loss + identity_A_loss) / 2
            epoch_identity_loss += identity_loss.item()

            # Total loss
            gen_loss = (valid_loss + cycle_loss_rate*cycle_loss +
                        identity_loss_rate*identity_loss)/3
            epoch_total_gen_loss += gen_loss.item()

            gen_loss_unit = (valid_loss.clone().detach(
            ) + cycle_loss.clone().detach() + identity_loss.clone().detach())/3
            epoch_total_gen_loss_unit += gen_loss_unit.item()

            optimizer_gen_B.zero_grad()
            optimizer_gen_A.zero_grad()

            gen_loss.backward()
            if flag is True:
                gradFlowChecker.plotGrad(gen_A.named_parameters(), title='gen_A')
                gradFlowChecker.plotGrad(gen_B.named_parameters(), title='gen_B')

            optimizer_gen_B.step()
            optimizer_gen_A.step()

            # ========================================================================
            # Train discriminators with generators' parameters flozen
            # ========================================================================
            # loss for native image
            discriminated_A = dec_A(img_A)
            discriminated_B = dec_B(img_B)

            discriminated_A_label = torch.ones_like(discriminated_A).to(device)
            discriminated_B_label = torch.ones_like(discriminated_B).to(device)

            D_A_loss_1 = D_A_Loss_1(discriminated_A, discriminated_A_label)
            D_B_loss_1 = D_B_Loss_1(discriminated_B, discriminated_B_label)

            # loss for generated image
            generated_A = gen_A(img_B.detach())
            generated_B = gen_B(img_A.detach())

            valid_A = dec_A(generated_A)
            valid_B = dec_B(generated_B)
            valid_A_label = torch.zeros_like(valid_A).to(device)
            valid_B_label = torch.zeros_like(valid_B).to(device)

            D_A_loss_2 = D_A_Loss_2(valid_A, valid_A_label)  # (dec_gan_A, 0)
            D_B_loss_2 = D_B_Loss_2(valid_B, valid_B_label)  # (dec_gan_B, 0)

            D_gen_loss = (D_A_loss_2 + D_B_loss_2) / 2
            epoch_D_gen_loss += D_gen_loss

            # Total dec loss
            dec_loss = (D_gen_loss + D_A_loss_1 + D_B_loss_1) / 3
            epoch_total_dec_loss += dec_loss

            optimizer_dec_A.zero_grad()
            optimizer_dec_B.zero_grad()

            dec_loss.backward()
            if flag is True:
                gradFlowChecker.plotGrad(dec_A.named_parameters(), title='dec_A')
                gradFlowChecker.plotGrad(dec_B.named_parameters(), title='dec_B')
                flag = False

            optimizer_dec_A.step()
            optimizer_dec_B.step()

            # Scheduler

            scheduler_gen_B.step()
            scheduler_gen_A.step()
            scheduler_dec_A.step()
            scheduler_dec_B.step()

            optimizer_gen_B.zero_grad()
            optimizer_gen_A.zero_grad()
            optimizer_dec_A.zero_grad()
            optimizer_dec_B.zero_grad()

        with torch.no_grad():
            logs[0][epoch] = epoch_total_dec_loss
            logs[1][epoch] = epoch_total_gen_loss

            epoch_finish_time = time.perf_counter()
            calc_time = epoch_finish_time - epoch_start_time

            print(
                '----------------------------------------------------------------------------')
            print('epoch_adversarial_loss: {:.4f}   |   epoch_cycle_consistency_loss: {:.4f}   |   epoch_identity_loss: {:.4f}'.format(
                epoch_valid_loss, epoch_cycle_loss, epoch_identity_loss))
            print('epoch_total_gen_loss: {:.4f}'.format(epoch_total_gen_loss))
            print(
                '----------------------------------------------------------------------------')
            print('epoch_D_gen_loss: {:.4f}   |   epoch_total_dec_loss: {:.4f}'.format(
                epoch_D_gen_loss, epoch_total_dec_loss))
            print(
                '----------------------------------------------------------------------------')
            print('Time: {:.4f}[sec]'.format(calc_time))
            print('')

            image_saver(gen_B=gen_B, gen_A=gen_A, img_A=img_A,
                        img_B=img_B, batch_size=batch_size, epoch=epoch)
            img_path = './pictures/img 7.jpeg'
            epoch_image_saver(gen_A=gen_A, img_path=img_path,
                              epoch=epoch, device=device)

    weight_saver(gen_B=gen_B, gen_A=gen_A)

    losses = [epoch_total_gen_loss_unit, epoch_total_dec_loss]
    if toReturnLoss:
        return gen_B, gen_A, dec_A, dec_B, logs, losses
    return gen_B, gen_A, dec_A, dec_B, logs
