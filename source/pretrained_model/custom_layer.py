

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parameters import PRETRAINED_MODEL_PATH
import torch
import torch.nn as nn



def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = torch.div(num_channels, groups, rounding_mode='floor')

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width) # type: ignore

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class BatchNormReLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.batchnorm = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class FireBlock_v1(nn.Module):
    '''
    @date: 2023.03.15
    @description: squeeze and 1-path expand
    '''
    def __init__(self, input_channels, squeeze_channels, expand_channels, groups):
        super().__init__()
        # squeeze
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)
        self.squeeze_bn_relu = BatchNormReLU(squeeze_channels)
        # expand
        self.expand_3 = nn.Conv2d(in_channels=squeeze_channels, out_channels=expand_channels, kernel_size=(3, 3), padding='same', groups=groups)
        self.expand_3_bn_relu = BatchNormReLU(expand_channels)

    def forward(self, x):
        s = self.squeeze(x)
        s = self.squeeze_bn_relu(s)
        e_3 = self.expand_3(s)
        e_3 = self.expand_3_bn_relu(e_3)
        return e_3

class FireBlock_v2(nn.Module):
    '''
    @date: 2023.03.18
    @description: squeeze and 2-path expand
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, groups):
        super().__init__()
        # squeeze
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)
        self.squeeze_bn_relu = BatchNormReLU(squeeze_channels)
        # expand
        self.expand_1 = nn.Conv2d(in_channels=squeeze_channels, out_channels=e_1_channels, kernel_size=(1, 1))
        self.expand_1_bn_relu = BatchNormReLU(e_1_channels)
        self.expand_3 = nn.Conv2d(in_channels=squeeze_channels, out_channels=e_3_channels, kernel_size=(3, 3), padding='same', groups=groups)
        self.expand_3_bn_relu = BatchNormReLU(e_3_channels)

    def forward(self, x):
        s = self.squeeze(x)
        s = self.squeeze_bn_relu(s)
        e_1 = self.expand_1(s)
        e_1 = self.expand_1_bn_relu(e_1)
        e_3 = self.expand_3(s)
        e_3 = self.expand_3_bn_relu(e_3)
        x = torch.cat([e_1, e_3], dim=1)
        return x

class FireBlock_v3(nn.Module):
    '''
    @date: 2023.03.19
    @description: squeeze and expand, designed for 996*166 not cut model
                  add Asymmetric Convolution，1*3 & 3*1, which equals to 3*1 & 1*3
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, e_5_channels, groups):
        super().__init__()
        # squeeze
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)
        self.squeeze_bn_relu = BatchNormReLU(squeeze_channels)
        # expand
        self.expand_1 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_1_channels, kernel_size=(1, 1), padding='same', groups=groups),
            BatchNormReLU(e_1_channels)
        )
        self.expand_3 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_3_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(1, 3), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(3, 1), padding='same', groups=groups),
            BatchNormReLU(e_3_channels)
        )
        self.expand_5 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_5_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(1, 3), padding='same', dilation=2, groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(3, 1), padding='same', dilation=2, groups=groups),
            BatchNormReLU(e_5_channels)
        )

    def forward(self, x):
        s = self.squeeze(x)
        s = self.squeeze_bn_relu(s)
        e_1 = self.expand_1(s)
        e_3 = self.expand_3(s)
        e_5 = self.expand_5(s)
        x = torch.cat([e_1, e_3, e_5], dim=1)
        return x

class FireBlock_v4(nn.Module):
    '''
    @date: 2023.03.21
    @description: squeeze and expand, designed for 996*166 not cut model
                  add Asymmetric Convolution，1*3 & 3*1, which equals to 3*1 & 1*3
                  based on v3,
                  remove the BatchNormReLU in the squeeze & expand layers, only add a BatchNormReLU in the output
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, e_5_channels, groups):
        super().__init__()
        # squeeze
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)
        # expand
        self.expand_1 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_1_channels, kernel_size=(1, 1), padding='same', groups=groups),
        )
        self.expand_3 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_3_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(1, 3), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(3, 1), padding='same', groups=groups),
        )
        self.expand_5 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_5_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(1, 3), padding='same', dilation=2, groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(3, 1), padding='same', dilation=2, groups=groups),
        )
        self.bnr = BatchNormReLU(e_1_channels+e_3_channels+e_5_channels)

    def forward(self, x):
        s = self.squeeze(x)
        e_1 = self.expand_1(s)
        e_3 = self.expand_3(s)
        e_5 = self.expand_5(s)
        x = torch.cat([e_1, e_3, e_5], dim=1)
        x = self.bnr(x)
        return x

class FireBlock_v5(nn.Module):
    '''
    @date: 2023.03.26
    @description: squeeze and expand, designed for 996*166 not cut model
                  add Asymmetric Convolution，1*3 & 3*1, which equals to 3*1 & 1*3
                  based on v3,
                  replace BatchNorm with LayerNorm
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, e_5_channels, ln_size, groups):
        super().__init__()
        # squeeze
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.LayerNorm([squeeze_channels, ln_size[0], ln_size[1]]),
            nn.ReLU()
        )
        # expand
        self.expand_1 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_1_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.LayerNorm([e_1_channels, ln_size[0], ln_size[1]]),
            nn.ReLU()
        )
        self.expand_3 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_3_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(1, 3), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(3, 1), padding='same', groups=groups),
            nn.LayerNorm([e_3_channels, ln_size[0], ln_size[1]]),
            nn.ReLU()
        )
        self.expand_5 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_5_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(1, 3), padding='same', dilation=2, groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(3, 1), padding='same', dilation=2, groups=groups),
            nn.LayerNorm([e_5_channels, ln_size[0], ln_size[1]]),
            nn.ReLU()
        )

    def forward(self, x):
        s = self.squeeze(x)
        e_1 = self.expand_1(s)
        e_3 = self.expand_3(s)
        e_5 = self.expand_5(s)
        x = torch.cat([e_1, e_3, e_5], dim=1)
        return x

class FireBlock_v6(nn.Module):
    '''
    @date: 2023.03.19
    @description: squeeze and expand, designed for 996*166 not cut model
                  add Asymmetric Convolution，1*3 & 3*1, which equals to 3*1 & 1*3
                  based on FireBlock_v3
                  replace kernel 1*3 & dilation 2 with kernel 1*5
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, e_5_channels, groups):
        super().__init__()
        # squeeze
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)
        self.squeeze_bn_relu = BatchNormReLU(squeeze_channels)
        # expand
        self.expand_1 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_1_channels, kernel_size=(1, 1), padding='same', groups=groups),
            BatchNormReLU(e_1_channels)
        )
        self.expand_3 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_3_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(1, 3), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(3, 1), padding='same', groups=groups),
            BatchNormReLU(e_3_channels)
        )
        self.expand_5 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_5_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(1, 5), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(5, 1), padding='same', groups=groups),
            BatchNormReLU(e_5_channels)
        )

    def forward(self, x):
        s = self.squeeze(x)
        s = self.squeeze_bn_relu(s)
        e_1 = self.expand_1(s)
        e_3 = self.expand_3(s)
        e_5 = self.expand_5(s)
        x = torch.cat([e_1, e_3, e_5], dim=1)
        return x

class FireBlock_v7(nn.Module):
    '''
    @date: 2023.03.28
    @description: squeeze and expand, designed for 996*166 not cut model
                  add Asymmetric Convolution，1*3 & 3*1, which equals to 3*1 & 1*3
                  based on FireBlock_v3
                  init 1*3 3*1 weight bias with pretraind layer
    '''
    def __init__(self, input_channels, squeeze_channels, e_1_channels, e_3_channels, e_5_channels, groups):
        super().__init__()
        # squeeze
        self.squeeze = nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=(1, 1), padding='same', groups=groups)
        self.squeeze_bn_relu = BatchNormReLU(squeeze_channels)
        # expand
        self.expand_1 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_1_channels, kernel_size=(1, 1), padding='same', groups=groups),
            BatchNormReLU(e_1_channels)
        )
        self.expand_3 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_3_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(1, 3), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_3_channels, out_channels=e_3_channels, kernel_size=(3, 1), padding='same', groups=groups),
            BatchNormReLU(e_3_channels)
        )
        self.expand_5 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=e_5_channels, kernel_size=(1, 1), padding='same', groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(1, 3), padding='same', dilation=2, groups=groups),
            nn.Conv2d(in_channels=e_5_channels, out_channels=e_5_channels, kernel_size=(3, 1), padding='same', dilation=2, groups=groups),
            BatchNormReLU(e_5_channels)
        )

        self._init_by_pretrained_layers()

    def _init_by_pretrained_layers(self):
        checkpoint = torch.load(PRETRAINED_MODEL_PATH+r'/pretrained_layer/model/PretrainedLayers_best.pth')
        saved_dict = checkpoint['net']
        expand_3_state_dict = self.expand_3.state_dict()
        for param_tensor_name in expand_3_state_dict:
            weight_name, bias_name = get_init_layer_name(expand_3_state_dict[param_tensor_name].size())
            if weight_name:
                expand_3_state_dict[param_tensor_name] = saved_dict[weight_name]
                expand_3_state_dict[param_tensor_name.replace('weight', 'bias')] = saved_dict[bias_name]
        self.expand_3.load_state_dict(expand_3_state_dict)
        expand_5_state_dict = self.expand_5.state_dict()
        for param_tensor_name in expand_5_state_dict:
            weight_name, bias_name = get_init_layer_name(expand_5_state_dict[param_tensor_name].size())
            if weight_name:
                expand_5_state_dict[param_tensor_name] = saved_dict[weight_name]
                expand_5_state_dict[param_tensor_name.replace('weight', 'bias')] = saved_dict[bias_name]
        self.expand_5.load_state_dict(expand_5_state_dict)

    def forward(self, x):
        s = self.squeeze(x)
        s = self.squeeze_bn_relu(s)
        e_1 = self.expand_1(s)
        e_3 = self.expand_3(s)
        e_5 = self.expand_5(s)
        x = torch.cat([e_1, e_3, e_5], dim=1)
        return x

class SEBlock_v1(nn.Module):
    '''
    @date: 2023.03.15
    @description: squeeze and excitation
    '''
    def __init__(self, h_channels, reduction):
        super().__init__()
        self.sequential = nn.Sequential(
            # squeeze
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            # excitation
            nn.Linear(h_channels, reduction),
            nn.BatchNorm1d(reduction),
            nn.ReLU(),
            nn.Linear(reduction, h_channels),
            nn.BatchNorm1d(h_channels),
            nn.Sigmoid()
        )

    def forward(self, h):
        s = self.sequential(h)
        # squeeze & excitation
        s = s.reshape(s.shape[0], s.shape[1], 1, 1)
        # scale
        x = torch.mul(h, s)
        return x

class SEBlock_v2(nn.Module):
    '''
    @date: 2023.03.21
    @description: squeeze and excitation
                  this block is used for get attention, so remove the ReLU, just use linear
    '''
    def __init__(self, h_channels, reduction):
        super().__init__()
        self.sequential = nn.Sequential(
            # squeeze
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            # excitation
            nn.BatchNorm1d(h_channels),
            nn.Linear(h_channels, reduction),
            nn.Linear(reduction, h_channels),
            nn.Sigmoid()
        )

    def forward(self, h):
        s = self.sequential(h)
        # squeeze & excitation
        s = s.reshape(s.shape[0], s.shape[1], 1, 1)
        # scale
        x = torch.mul(h, s)
        return x

class SKBlock_v1(nn.Module):
    '''
    @date: 2023.03.15
    @description: split, fuse and select
    '''
    def __init__(self, h_channels, out_channels, reduction):
        super().__init__()
        self.u_1 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            BatchNormReLU(out_channels)
        )
        self.u_3 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
            BatchNormReLU(out_channels)
        )
        self.u_5 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same', dilation=2),
            BatchNormReLU(out_channels)
        )
        self.sequential = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_channels, reduction),
            nn.BatchNorm1d(reduction),
            nn.ReLU()
        )
        self.u_a = nn.Linear(reduction, out_channels)
        self.u_b = nn.Linear(reduction, out_channels)
        self.u_c = nn.Linear(reduction, out_channels)

    def forward(self, h):
        # 1 split
        u_1 = self.u_1(h)
        u_3 = self.u_3(h)
        u_5 = self.u_5(h)
        # 2 fuse
        # 2.1 integrate information from all branches.
        u = u_1 + u_3 + u_5
        # 2.2 global average pooling.
        # 2.3 compact feature by simple fully connected (fc) layer.
        z = self.sequential(u)
        # 3 select
        # 3.1 Soft attention across channels
        u_a = self.u_a(z)
        u_b = self.u_b(z)
        u_c = self.u_c(z)
        u_abc = nn.Softmax(dim=1)(torch.stack((u_a, u_b, u_c), dim=1))
        u_a, u_b, u_c = torch.split(u_abc, split_size_or_sections=1, dim=1)
        u_a = u_a.reshape(u_a.shape[0], u_a.shape[2], 1, 1)
        u_b = u_b.reshape(u_b.shape[0], u_b.shape[2], 1, 1)
        u_c = u_c.reshape(u_c.shape[0], u_c.shape[2], 1, 1)
        # 3.2 The final feature map V is obtained through the attention weights on various kernels.
        v = torch.mul(u_1, u_a) + torch.mul(u_3, u_b) + torch.mul(u_5, u_c)
        return v

class SKBlock_v2(nn.Module):
    '''
    @date: 2023.03.19
    @description: split, fuse and select
    '''
    def __init__(self, h_channels, out_channels, reduction):
        super().__init__()
        self.u_1 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            BatchNormReLU(out_channels)
        )
        self.u_3 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 3), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), padding='same'),
            BatchNormReLU(out_channels)
        )
        self.u_5 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 3), padding='same', dilation=2),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), padding='same', dilation=2),
            BatchNormReLU(out_channels)
        )
        self.sequential = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_channels, reduction),
            nn.BatchNorm1d(reduction),
            nn.ReLU()
        )
        self.u_a = nn.Linear(reduction, out_channels)
        self.u_b = nn.Linear(reduction, out_channels)
        self.u_c = nn.Linear(reduction, out_channels)

    def forward(self, h):
        # 1 split
        u_1 = self.u_1(h)
        u_3 = self.u_3(h)
        u_5 = self.u_5(h)
        # 2 fuse
        # 2.1 integrate information from all branches.
        u = u_1 + u_3 + u_5
        # 2.2 global average pooling.
        # 2.3 compact feature by simple fully connected (fc) layer.
        z = self.sequential(u)
        # 3 select
        # 3.1 Soft attention across channels
        u_a = self.u_a(z)
        u_b = self.u_b(z)
        u_c = self.u_c(z)
        u_abc = nn.Softmax(dim=1)(torch.stack((u_a, u_b, u_c), dim=1))
        u_a, u_b, u_c = torch.split(u_abc, split_size_or_sections=1, dim=1)
        u_a = u_a.reshape(u_a.shape[0], u_a.shape[2], 1, 1)
        u_b = u_b.reshape(u_b.shape[0], u_b.shape[2], 1, 1)
        u_c = u_c.reshape(u_c.shape[0], u_c.shape[2], 1, 1)
        # 3.2 The final feature map V is obtained through the attention weights on various kernels.
        v = torch.mul(u_1, u_a) + torch.mul(u_3, u_b) + torch.mul(u_5, u_c)
        return v

class SKBlock_v3(nn.Module):
    '''
    @date: 2023.03.21
    @description: split, fuse and select
                  based on v2,
                  cancel the BatchNormReLU in select part, add the BatchNormReLU after select part
    '''
    def __init__(self, h_channels, out_channels, reduction):
        super().__init__()
        self.u_1 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same')
        )
        self.u_3 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 3), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), padding='same')
        )
        self.u_5 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 3), padding='same', dilation=2),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), padding='same', dilation=2)
        )
        self.sequential = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_channels, reduction),
            nn.BatchNorm1d(reduction),
            nn.ReLU()
        )
        self.u_a = nn.Linear(reduction, out_channels)
        self.u_b = nn.Linear(reduction, out_channels)
        self.u_c = nn.Linear(reduction, out_channels)
        self.bnr = BatchNormReLU(out_channels)

    def forward(self, h):
        # 1 split
        u_1 = self.u_1(h)
        u_3 = self.u_3(h)
        u_5 = self.u_5(h)
        # 2 fuse
        # 2.1 integrate information from all branches.
        u = u_1 + u_3 + u_5
        # 2.2 global average pooling.
        # 2.3 compact feature by simple fully connected (fc) layer.
        z = self.sequential(u)
        # 3 select
        # 3.1 Soft attention across channels
        u_a = self.u_a(z)
        u_b = self.u_b(z)
        u_c = self.u_c(z)
        u_abc = nn.Softmax(dim=1)(torch.stack((u_a, u_b, u_c), dim=1))
        u_a, u_b, u_c = torch.split(u_abc, split_size_or_sections=1, dim=1)
        u_a = u_a.reshape(u_a.shape[0], u_a.shape[2], 1, 1)
        u_b = u_b.reshape(u_b.shape[0], u_b.shape[2], 1, 1)
        u_c = u_c.reshape(u_c.shape[0], u_c.shape[2], 1, 1)
        # 3.2 The final feature map V is obtained through the attention weights on various kernels.
        v = torch.mul(u_1, u_a) + torch.mul(u_3, u_b) + torch.mul(u_5, u_c)
        v = self.bnr(v)
        return v

class SKBlock_v4(nn.Module):
    '''
    @date: 2023.03.26
    @description: split, fuse and select
                  based on v2,
                  replace BatchNorm with LayerNorm
    '''
    def __init__(self, h_channels, out_channels, reduction, ln_size):
        super().__init__()
        self.u_1 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            nn.LayerNorm([out_channels, ln_size[0], ln_size[1]]),
            nn.ReLU()
        )
        self.u_3 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 3), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), padding='same'),
            nn.LayerNorm([out_channels, ln_size[0], ln_size[1]]),
            nn.ReLU()
        )
        self.u_5 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 3), padding='same', dilation=2),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), padding='same', dilation=2),
            nn.LayerNorm([out_channels, ln_size[0], ln_size[1]]),
            nn.ReLU()
        )
        self.sequential = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_channels, reduction),
            nn.BatchNorm1d(reduction),
            nn.ReLU()
        )
        self.u_a = nn.Linear(reduction, out_channels)
        self.u_b = nn.Linear(reduction, out_channels)
        self.u_c = nn.Linear(reduction, out_channels)

    def forward(self, h):
        # 1 split
        u_1 = self.u_1(h)
        u_3 = self.u_3(h)
        u_5 = self.u_5(h)
        # 2 fuse
        # 2.1 integrate information from all branches.
        u = u_1 + u_3 + u_5
        # 2.2 global average pooling.
        # 2.3 compact feature by simple fully connected (fc) layer.
        z = self.sequential(u)
        # 3 select
        # 3.1 Soft attention across channels
        u_a = self.u_a(z)
        u_b = self.u_b(z)
        u_c = self.u_c(z)
        u_abc = nn.Softmax(dim=1)(torch.stack((u_a, u_b, u_c), dim=1))
        u_a, u_b, u_c = torch.split(u_abc, split_size_or_sections=1, dim=1)
        u_a = u_a.reshape(u_a.shape[0], u_a.shape[2], 1, 1)
        u_b = u_b.reshape(u_b.shape[0], u_b.shape[2], 1, 1)
        u_c = u_c.reshape(u_c.shape[0], u_c.shape[2], 1, 1)
        # 3.2 The final feature map V is obtained through the attention weights on various kernels.
        v = torch.mul(u_1, u_a) + torch.mul(u_3, u_b) + torch.mul(u_5, u_c)
        return v

class SKBlock_v5(nn.Module):
    '''
    @date: 2023.03.19
    @description: split, fuse and select
                  based on SKBlock_v2,
                  replace kernel 1*3 & dilation 2 with kernel 1*5
    '''
    def __init__(self, h_channels, out_channels, reduction):
        super().__init__()
        self.u_1 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            BatchNormReLU(out_channels)
        )
        self.u_3 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 3), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), padding='same'),
            BatchNormReLU(out_channels)
        )
        self.u_5 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 5), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(5, 1), padding='same'),
            BatchNormReLU(out_channels)
        )
        self.sequential = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_channels, reduction),
            nn.BatchNorm1d(reduction),
            nn.ReLU()
        )
        self.u_a = nn.Linear(reduction, out_channels)
        self.u_b = nn.Linear(reduction, out_channels)
        self.u_c = nn.Linear(reduction, out_channels)

    def forward(self, h):
        # 1 split
        u_1 = self.u_1(h)
        u_3 = self.u_3(h)
        u_5 = self.u_5(h)
        # 2 fuse
        # 2.1 integrate information from all branches.
        u = u_1 + u_3 + u_5
        # 2.2 global average pooling.
        # 2.3 compact feature by simple fully connected (fc) layer.
        z = self.sequential(u)
        # 3 select
        # 3.1 Soft attention across channels
        u_a = self.u_a(z)
        u_b = self.u_b(z)
        u_c = self.u_c(z)
        u_abc = nn.Softmax(dim=1)(torch.stack((u_a, u_b, u_c), dim=1))
        u_a, u_b, u_c = torch.split(u_abc, split_size_or_sections=1, dim=1)
        u_a = u_a.reshape(u_a.shape[0], u_a.shape[2], 1, 1)
        u_b = u_b.reshape(u_b.shape[0], u_b.shape[2], 1, 1)
        u_c = u_c.reshape(u_c.shape[0], u_c.shape[2], 1, 1)
        # 3.2 The final feature map V is obtained through the attention weights on various kernels.
        v = torch.mul(u_1, u_a) + torch.mul(u_3, u_b) + torch.mul(u_5, u_c)
        return v

class SKBlock_v6(nn.Module):
    '''
    @date: 2023.03.28
    @description: split, fuse and select,
                  based on SKBlock_v2,
                  init 1*3 3*1 weight bias with pretraind layer
    '''
    def __init__(self, h_channels, out_channels, reduction):
        super().__init__()
        self.u_1 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            BatchNormReLU(out_channels)
        )
        self.u_3 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 3), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), padding='same'),
            BatchNormReLU(out_channels)
        )
        self.u_5 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=out_channels, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 3), padding='same', dilation=2),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), padding='same', dilation=2),
            BatchNormReLU(out_channels)
        )
        self.sequential = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_channels, reduction),
            nn.BatchNorm1d(reduction),
            nn.ReLU()
        )
        self.u_a = nn.Linear(reduction, out_channels)
        self.u_b = nn.Linear(reduction, out_channels)
        self.u_c = nn.Linear(reduction, out_channels)

        self._init_by_pretrained_layers()

    def _init_by_pretrained_layers(self):
        checkpoint = torch.load(PRETRAINED_MODEL_PATH+r'/pretrained_layer/model/PretrainedLayers_best.pth')
        saved_dict = checkpoint['net']

        u_3_state_dict = self.u_3.cuda().state_dict()
        for origin_param_name in u_3_state_dict:
            weight_name, bias_name = get_init_layer_name(u_3_state_dict[origin_param_name].size())
            if weight_name:
                u_3_state_dict[origin_param_name] = 0.5*saved_dict[weight_name] + 0.5*u_3_state_dict[origin_param_name]
                u_3_state_dict[origin_param_name.replace('weight', 'bias')] = 0.5*saved_dict[bias_name] + 0.5*u_3_state_dict[origin_param_name.replace('weight', 'bias')]
        self.u_3.load_state_dict(u_3_state_dict)

        u_5_state_dict = self.u_5.cuda().state_dict()
        for origin_param_name in u_5_state_dict:
            weight_name, bias_name = get_init_layer_name(u_5_state_dict[origin_param_name].size())
            if weight_name:
                u_5_state_dict[origin_param_name] = 0.5*saved_dict[weight_name] + 0.5*u_5_state_dict[origin_param_name]
                u_5_state_dict[origin_param_name.replace('weight', 'bias')] = 0.5*saved_dict[bias_name] + 0.5*u_5_state_dict[origin_param_name.replace('weight', 'bias')]
        self.u_5.load_state_dict(u_5_state_dict)
        

    def forward(self, h):
        # 1 split
        u_1 = self.u_1(h)
        u_3 = self.u_3(h)
        u_5 = self.u_5(h)
        # 2 fuse
        # 2.1 integrate information from all branches.
        u = u_1 + u_3 + u_5
        # 2.2 global average pooling.
        # 2.3 compact feature by simple fully connected (fc) layer.
        z = self.sequential(u)
        # 3 select
        # 3.1 Soft attention across channels
        u_a = self.u_a(z)
        u_b = self.u_b(z)
        u_c = self.u_c(z)
        u_abc = nn.Softmax(dim=1)(torch.stack((u_a, u_b, u_c), dim=1))
        u_a, u_b, u_c = torch.split(u_abc, split_size_or_sections=1, dim=1)
        u_a = u_a.reshape(u_a.shape[0], u_a.shape[2], 1, 1)
        u_b = u_b.reshape(u_b.shape[0], u_b.shape[2], 1, 1)
        u_c = u_c.reshape(u_c.shape[0], u_c.shape[2], 1, 1)
        # 3.2 The final feature map V is obtained through the attention weights on various kernels.
        v = torch.mul(u_1, u_a) + torch.mul(u_3, u_b) + torch.mul(u_5, u_c)
        return v

class SpatialAttentionMapBlock_v1(nn.Module):
    '''
    @date: 2023.03.20
    @description: a block for getting Spatial Attention Map of SAOL (Spatially Attentive Output Layer)
    '''
    def __init__(self, h, w, in_c, mid_c):
        '''
        @params: h: height of feature map
                 w: width  of feature map
        '''
        super().__init__()

        self.sam_block = nn.Sequential(
            nn.AdaptiveMaxPool2d((h, w)),
            nn.Conv2d(in_channels=in_c, out_channels=mid_c, kernel_size=(3, 2), padding='same'),
            BatchNormReLU(mid_c),
            nn.Conv2d(in_channels=mid_c, out_channels=1, kernel_size=(3, 2), padding='same')
        )

    def forward(self, h):
        h = self.sam_block(h)
        # 扁平化输入张量, flatten the 2d feature map to 1d
        flat_input_tensor = h.reshape(h.shape[0], h.shape[1], -1)
        # 使用dim=2进行softmax
        softmax_output = nn.Softmax(dim=2)(flat_input_tensor)
        # 把形状变回 h 的样子
        sam = softmax_output.reshape(h.shape)
        return sam

class SpatialLogitsBlock_v1(nn.Module):
    '''
    @date: 2023.03.20
    @description: a block for getting Spatial Logits of SAOL (Spatially Attentive Output Layer)
    '''
    def __init__(self, h, w, in_c, mid_c, out_c):
        super().__init__()
        assert type(in_c) == tuple and len(in_c) == 3
        
        self.resize_1_block = nn.Sequential(
            nn.AdaptiveMaxPool2d((h, w)),
            nn.Conv2d(in_channels=in_c[0], out_channels=mid_c, kernel_size=(1, 1), padding='same')
        )
        self.resize_2_block = nn.Sequential(
            nn.AdaptiveMaxPool2d((h, w)),
            nn.Conv2d(in_channels=in_c[1], out_channels=mid_c, kernel_size=(1, 1), padding='same')
        )
        self.resize_3_block = nn.Sequential(
            nn.AdaptiveMaxPool2d((h, w)),
            nn.Conv2d(in_channels=in_c[2], out_channels=mid_c, kernel_size=(1, 1), padding='same')
        )
        self.sl_out_block = nn.Sequential(
            nn.BatchNorm2d(mid_c*3),
            nn.Conv2d(in_channels=mid_c*3, out_channels=out_c, kernel_size=(3, 2), padding='same'),
            nn.Softmax(dim=1)
        )

    def forward(self, h1, h2, h3):
        h1 = self.resize_1_block(h1)
        h2 = self.resize_2_block(h2)
        h3 = self.resize_3_block(h3) # h1 h2 h3 output shape: (batch_size, mid_c, h, w)
        cat_h = torch.cat((h1, h2, h3), dim=1) # cat_h shape: (batch_size, 3*mid_c, h, w)
        sl = self.sl_out_block(cat_h)
        return sl

def get_init_layer_name(weight_shape):
    weightshape2layername = {torch.Size([16, 16, 1, 3]): 'conv_3.1.weight',
                             torch.Size([16, 16, 3, 1]): 'conv_3.2.weight',
                             torch.Size([32, 32, 1, 3]): 'conv_3.7.weight',
                             torch.Size([32, 32, 3, 1]): 'conv_3.8.weight',
                             torch.Size([64, 64, 1, 3]): 'conv_3.13.weight',
                             torch.Size([64, 64, 3, 1]): 'conv_3.14.weight',
                             torch.Size([128, 128, 1, 3]): 'conv_3.19.weight',
                             torch.Size([128, 128, 3, 1]): 'conv_3.20.weight',
                             torch.Size([16, 16, 1, 5]): 'conv_5.1.weight',
                             torch.Size([16, 16, 5, 1]): 'conv_5.2.weight',
                             torch.Size([32, 32, 1, 5]): 'conv_5.7.weight',
                             torch.Size([32, 32, 5, 1]): 'conv_5.8.weight',
                             torch.Size([64, 64, 1, 5]): 'conv_5.13.weight',
                             torch.Size([64, 64, 5, 1]): 'conv_5.14.weight',
                             torch.Size([128, 128, 1, 5]): 'conv_5.19.weight',
                             torch.Size([128, 128, 5, 1]): 'conv_5.20.weight'}
    if weight_shape in weightshape2layername.keys():
        # print(f"weight_shape: {weight_shape} is in weightshape2layername.keys()")
        layerweightname = weightshape2layername[weight_shape]
        layerbiasname = layerweightname.replace('weight', 'bias')
        return layerweightname, layerbiasname
    else:
        # print(f"weight_shape: {weight_shape} is not in weightshape2layername.keys()")
        return False, False

class AttentionPool_v1(nn.Module):
    '''
    @description: attention & pool on width or height
    '''
    def __init__(self, attention_dim, attention_len, out_w, out_h, reduction_dim):
        '''
        @param attention_dim: 2 or 3, 2 means attention & pool on width, 3 means attention & pool on height
        '''
        super().__init__()
        assert attention_dim in [2,3]
        self.attention_dim = attention_dim
        self.out_w = out_w
        self.out_h = out_h
        self.attention_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(attention_len, reduction_dim),
            nn.BatchNorm1d(reduction_dim),
            nn.ReLU(),
            nn.Linear(reduction_dim, attention_len),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        assert len(x.size()) == 4, "fit for 4-dim [N, C, W, H] tensor"
        if self.attention_dim == 2:
            # assert self.out_h == x.size()[3]
            attention_vector = torch.permute(x, (0, 2, 1, 3))
            attention_vector = self.attention_layer(attention_vector)
            _, max_indices = torch.topk(attention_vector, self.out_w, dim=1)
            max_values_sorted, _ = torch.sort(max_indices, dim=1)
            pooled_x = torch.stack([x[i, :, max_values_sorted[i], :] for i in range(x.size()[0])], dim=0)
        else:
            # assert self.out_w == x.size()[2]
            attention_vector = torch.permute(x, (0, 3, 1, 2))
            attention_vector = self.attention_layer(attention_vector)
            _, max_indices = torch.topk(attention_vector, self.out_h, dim=1)
            max_values_sorted, _ = torch.sort(max_indices, dim=1)
            pooled_x = torch.stack([x[i, :,:, max_values_sorted[i]] for i in range(x.size()[0])], dim=0)
        return pooled_x
        

if __name__ == "__main__":
    SKBlock_v6(h_channels=64, out_channels=64, reduction=32)

