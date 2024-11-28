from net.decoder_ori import *
from net.encoder_ori import *
from loss.distortion import Distortion
from net.channel_ori import Channel
from random import choice
import torch.nn as nn
from net.GDN import IGDN,GDN

class AFModule(nn.Module):
    def __init__(self, input_channels):
        super(AFModule, self).__init__()
        # self.name_prefix = name_prefix
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.concat_channels = input_channels + 1  # Assuming SNR is a single scalar value
        self.dense1 = nn.Linear(self.concat_channels, input_channels // 16)
        self.dense2 = nn.Linear(input_channels // 16, input_channels)
    
    def forward(self, inputs, snr):
        batch_size, channels, height, width = inputs.size()
        
        # Global Average Pooling
        x = self.global_pool(inputs).view(batch_size, -1)
        
        # Concatenate with SNR
        
        x = torch.cat((x, snr), dim=1)
        
        # Dense layers
        x = x.to(self.dense1.weight.dtype)
        x = F.relu(self.dense1(x))
        x = torch.sigmoid(self.dense2(x))
        
        # Reshape to match the input dimensions for multiplication
        x = x.view(batch_size, channels, 1, 1)
        
        # Multiply input by the output of the dense layers
        out = inputs * x
        
        return out

class WITT(nn.Module):
    def __init__(self, args, config, device):
        super(WITT, self).__init__()
        self.config = config
        self.device = device
        self.multiple_snr = [int(snr) for snr in args.multiple_snr.split(",")]
        self.downsample = config.downsample
        self.model = args.model

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=9, stride=2, padding=2),
            GDN(ch=256, device=device),
            nn.PReLU(num_parameters=256),
            AFModule(input_channels=256),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=3),
            GDN(ch=256, device=device),
            nn.PReLU(num_parameters=256),
            AFModule(input_channels=256),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2),
            GDN(ch=256, device=device),
            nn.PReLU(num_parameters=256),
            AFModule(input_channels=256),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2),
            GDN(ch=256, device=device),
            nn.PReLU(num_parameters=256),
            AFModule(input_channels=256),

            nn.Conv2d(in_channels=256, out_channels=args.C, kernel_size=5, stride=1, padding=2),
            GDN(ch=args.C, device=device),
        )

        # Decoder 1
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=args.C, out_channels=256, kernel_size=5, stride=1, padding=2),
            IGDN(ch=256, device=device),
            nn.PReLU(num_parameters=256),
            AFModule(input_channels=256),

            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2),
            IGDN(ch=256, device=device),
            nn.PReLU(num_parameters=256),
            AFModule(input_channels=256),

            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2),
            IGDN(ch=256, device=device),
            nn.PReLU(num_parameters=256),
            AFModule(input_channels=256),

            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=3, output_padding=1),
            IGDN(ch=256, device=device),
            nn.PReLU(num_parameters=256),
            AFModule(input_channels=256),

            nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=9, stride=2, padding=2, output_padding=1),
            IGDN(ch=3, device=device),
            nn.Sigmoid(),
        )

        # Decoder 2
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=args.C, out_channels=256, kernel_size=5, stride=1, padding=2),
            IGDN(ch=256, device=device),
            nn.PReLU(num_parameters=256),
            AFModule(input_channels=256),

            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2),
            IGDN(ch=256, device=device),
            nn.PReLU(num_parameters=256),
            AFModule(input_channels=256),

            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2),
            IGDN(ch=256, device=device),
            nn.PReLU(num_parameters=256),
            AFModule(input_channels=256),

            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=3, output_padding=1),
            IGDN(ch=256, device=device),
            nn.PReLU(num_parameters=256),
            AFModule(input_channels=256),

            nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=9, stride=2, padding=2, output_padding=1),
            IGDN(ch=3, device=device),
            nn.Sigmoid(),
        )

        # Loss and other components
        self.distortion_loss = Distortion(args)
        self.channel = Channel(args, config)
        self.pass_channel = config.pass_channel
        self.squared_difference = torch.nn.MSELoss(reduction='none')

    def distortion_loss_wrapper(self, x_gen, x_real):
        distortion_loss = self.distortion_loss.forward(x_gen, x_real, normalization=self.config.norm)
        return distortion_loss

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def forward_layer(self, layer, x, snr=None):
        if isinstance(layer, AFModule):
            # B, _, H, W = x.shape
            # snr_cuda = torch.tensor(snr, dtype=torch.float).to(self.device)
            # snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
            return layer(x, snr)
        return layer(x)

    
    def forward(self, input_image, given_SNR1=None, given_SNR2=None, train_flag=True):
        B, _, H, W = input_image.shape

        # 默认信噪比处理
        if given_SNR1 is None:
            SNR1 = choice(self.multiple_snr)
        else:
            SNR1 = given_SNR1

        if given_SNR2 is None:
            SNR2 = choice(self.multiple_snr)
        else:
            SNR2 = given_SNR2

        # 编码器提取共享特征
        feature = input_image
        for layer in self.encoder:
            feature = self.forward_layer(layer, feature, SNR1)  # 使用 SNR1 编码共享特征

        CBR = feature.numel() / 2 / input_image.numel()

        # 信道处理（选择是否通过信道传输）
        if self.pass_channel:
            noisy_feature1 = self.feature_pass_channel(feature, SNR1)
            noisy_feature2 = self.feature_pass_channel(feature, SNR2)
        else:
            noisy_feature1 = feature
            noisy_feature2 = feature

        # 解码器1处理 noisy_feature1
        decoded_image1 = noisy_feature1
        for layer in self.decoder1:
            decoded_image1 = self.forward_layer(layer, decoded_image1, SNR1)

        # 解码器2处理 noisy_feature2
        decoded_image2 = noisy_feature2
        for layer in self.decoder2:
            decoded_image2 = self.forward_layer(layer, decoded_image2, SNR2)

        if train_flag:
            # 计算训练损失
            loss_G1 = self.distortion_loss.forward(input_image, decoded_image1.clamp(0., 1.))
            loss_G2 = self.distortion_loss.forward(input_image, decoded_image2.clamp(0., 1.))

            return decoded_image1, decoded_image2, CBR, SNR1, SNR2, loss_G1.mean(), loss_G2.mean()
        else:
            # 计算评估指标
            mse1 = self.squared_difference(input_image * 255., decoded_image1.clamp(0., 1.) * 255.)
            mse2 = self.squared_difference(input_image * 255., decoded_image2.clamp(0., 1.) * 255.)
            loss_G1 = self.distortion_loss.forward(input_image, decoded_image1.clamp(0., 1.))
            loss_G2 = self.distortion_loss.forward(input_image, decoded_image2.clamp(0., 1.))

            return decoded_image1, decoded_image2, CBR, SNR1, SNR2, mse1.mean(), mse2.mean(), loss_G1.mean(), loss_G2.mean()
        

        #why don't lai ？ rulai