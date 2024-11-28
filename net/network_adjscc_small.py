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
    def __init__(self, args, config,device):
        super(WITT, self).__init__()
        self.config = config
        encoder_kwargs = config.encoder_kwargs
        decoder_kwargs = config.decoder_kwargs
        self.device =device
        self.channel_nums = 32
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=self.channel_nums,kernel_size=9,stride=2,padding=2),
            GDN(ch=self.channel_nums, device=device),
            nn.PReLU(num_parameters=self.channel_nums),
            AFModule(input_channels=self.channel_nums),

            nn.Conv2d(in_channels=self.channel_nums,out_channels=self.channel_nums,kernel_size=5,stride=2,padding=3),
            GDN(ch=self.channel_nums, device=device),
            nn.PReLU(num_parameters=self.channel_nums),
            AFModule(input_channels=self.channel_nums),

            nn.Conv2d(in_channels=self.channel_nums,out_channels=self.channel_nums,kernel_size=5,stride=1,padding=2),
            GDN(ch=self.channel_nums, device=device),
            nn.PReLU(num_parameters=self.channel_nums),
            AFModule(input_channels=self.channel_nums),

            nn.Conv2d(in_channels=self.channel_nums,out_channels=self.channel_nums,kernel_size=5,stride=1,padding=2),
            GDN(ch=self.channel_nums, device=device),
            nn.PReLU(num_parameters=self.channel_nums),
            AFModule(input_channels=self.channel_nums),

            nn.Conv2d(in_channels=self.channel_nums,out_channels=args.C,kernel_size=5,stride=1,padding=2),
            GDN(ch=args.C, device=device),
            # nn.PReLU(num_parameters=c),
            
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=args.C,out_channels=self.channel_nums,kernel_size=5,stride=1,padding=2),
            IGDN(ch=self.channel_nums, device=device),
            nn.PReLU(num_parameters=self.channel_nums),
            AFModule(input_channels=self.channel_nums),

            nn.ConvTranspose2d(in_channels=self.channel_nums,out_channels=self.channel_nums,kernel_size=5,stride=1,padding=2),
            IGDN(ch=self.channel_nums, device=device),
            nn.PReLU(num_parameters=self.channel_nums),
            AFModule(input_channels=self.channel_nums),

            nn.ConvTranspose2d(in_channels=self.channel_nums,out_channels=self.channel_nums,kernel_size=5,stride=1,padding=2),
            IGDN(ch=self.channel_nums, device=device),
            nn.PReLU(num_parameters=self.channel_nums),
            AFModule(input_channels=self.channel_nums),

            nn.ConvTranspose2d(in_channels=self.channel_nums,out_channels=self.channel_nums,kernel_size=5,stride=2,padding=3,output_padding=1),
            IGDN(ch=self.channel_nums, device=device),
            nn.PReLU(num_parameters=self.channel_nums),
            AFModule(input_channels=self.channel_nums),

            nn.ConvTranspose2d(in_channels=self.channel_nums,out_channels=3,kernel_size=9,stride=2,padding=2,output_padding=1),
            IGDN(ch=3, device=device),
            nn.Sigmoid(),
            
        )
        # if config.logger is not None:
        #     config.logger.info("Network config: ")
        #     config.logger.info("Encoder: ")
        #     config.logger.info(encoder_kwargs)
        #     config.logger.info("Decoder: ")
        #     config.logger.info(decoder_kwargs)
        self.distortion_loss = Distortion(args)
        self.channel = Channel(args, config)
        self.pass_channel = config.pass_channel
        self.squared_difference = torch.nn.MSELoss(reduction='none')
        self.H = self.W = 0
        self.multiple_snr = args.multiple_snr.split(",")
        for i in range(len(self.multiple_snr)):
            self.multiple_snr[i] = int(self.multiple_snr[i])
        self.downsample = config.downsample
        self.model = args.model    


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

    def forward(self, input_image, given_SNR = None,train_flag = True):
        B, _, H, W = input_image.shape

        # if H != self.H or W != self.W:
        #     self.encoder.update_resolution(H, W)
        #     self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
        #     self.H = H
        #     self.W = W

        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR

        # feature = self.encoder(input_image,chan_param)
        feature = input_image
        # chan_param = torch.tensor(chan_param).expand(B).to(self.device).view(B,-1)
        for layer in self.encoder:
            feature = self.forward_layer(layer, feature, chan_param)

        CBR = feature.numel() / 2 / input_image.numel()
        # Feature pass channel
        if self.pass_channel:
            noisy_feature = self.feature_pass_channel(feature, chan_param)
        else:
            noisy_feature = feature #
        # recon_image = self.decoder(noisy_feature)
        for layer in self.decoder:
            noisy_feature  = self.forward_layer(layer, noisy_feature, chan_param)
        recon_image =  noisy_feature
        if train_flag:
            loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.))
        
            return recon_image, CBR, chan_param, loss_G.mean()
        else:
            mse = self.squared_difference(input_image * 255., recon_image.clamp(0., 1.) * 255.)
            loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.))
        
            return recon_image, CBR, chan_param,  mse.mean(),loss_G.mean()
        
    # def forward(self, input_image, given_SNR1=None, given_SNR2=None, train_flag=True):
    #     B, _, H, W = input_image.shape

    #     # 默认信噪比处理
    #     if given_SNR1 is None:
    #         SNR1 = choice(self.multiple_snr)
    #     else:
    #         SNR1 = given_SNR1

    #     if given_SNR2 is None:
    #         SNR2 = choice(self.multiple_snr)
    #     else:
    #         SNR2 = given_SNR2

    #     # 编码器提取共享特征
    #     feature = input_image
    #     for layer in self.encoder:
    #         feature = self.forward_layer(layer, feature, SNR1)  # 使用 SNR1 编码共享特征

    #     CBR = feature.numel() / 2 / input_image.numel()

    #     # 信道处理（选择是否通过信道传输）
    #     if self.pass_channel:
    #         noisy_feature1 = self.feature_pass_channel(feature, SNR1)
    #         noisy_feature2 = self.feature_pass_channel(feature, SNR2)
    #     else:
    #         noisy_feature1 = feature
    #         noisy_feature2 = feature

    #     # 解码器1处理 noisy_feature1
    #     decoded_image1 = noisy_feature1
    #     for layer in self.decoder1:
    #         decoded_image1 = self.forward_layer(layer, decoded_image1, SNR1)

    #     # 解码器2处理 noisy_feature2
    #     decoded_image2 = noisy_feature2
    #     for layer in self.decoder2:
    #         decoded_image2 = self.forward_layer(layer, decoded_image2, SNR2)

    #     if train_flag:
    #         # 计算训练损失
    #         loss_G1 = self.distortion_loss.forward(input_image, decoded_image1.clamp(0., 1.))
    #         loss_G2 = self.distortion_loss.forward(input_image, decoded_image2.clamp(0., 1.))

    #         return decoded_image1, decoded_image2, CBR, SNR1, SNR2, loss_G1.mean(), loss_G2.mean()
    #     else:
    #         # 计算评估指标
    #         mse1 = self.squared_difference(input_image * 255., decoded_image1.clamp(0., 1.) * 255.)
    #         mse2 = self.squared_difference(input_image * 255., decoded_image2.clamp(0., 1.) * 255.)
    #         loss_G1 = self.distortion_loss.forward(input_image, decoded_image1.clamp(0., 1.))
    #         loss_G2 = self.distortion_loss.forward(input_image, decoded_image2.clamp(0., 1.))

    #         return decoded_image1, decoded_image2, CBR, SNR1, SNR2, mse1.mean(), mse2.mean(), loss_G1.mean(), loss_G2.mean()
        

        #why don't lai ？ rulai