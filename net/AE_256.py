import torch
import torch.nn as nn
import random
import numpy as np
import torch
import torch.nn.functional as F
from net.GDN import IGDN,GDN


def set_seed(seed):
    random.seed(seed)  # Python 内置的随机数生成器
    np.random.seed(seed)  # NumPy 随机数生成器
    torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
    torch.cuda.manual_seed(seed)  # 设置 PyTorch 在 GPU 上的随机种子，如果使用 GPU

# 设置随机种子为固定值，例如 42
set_seed(42)
# device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")


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
        snr = snr.view(batch_size, -1)  # Assuming snr is [batch_size, 1]
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
    
    
class NormalizationLayerE_complex(nn.Module):
    def __init__(self,P):
        super(NormalizationLayerE_complex,self).__init__()
        
        self.P=P

    def forward(self,z):
        # input 64,4,8,8
        bs,channels,h,w = z.shape
        # 将z变成一维复数向量，
        z = z.view(-1,2)
        real_tensor = z[:,0]
        imag_tensor = z[:,1]
        z_complex = torch.complex(real_tensor,imag_tensor)
        # k = bs*channels*h*w
        sqrt1 = torch.sqrt(torch.tensor(bs*channels*h*w*self.P))
        sqrt2 = torch.norm(z_complex)
        div = torch.div(z_complex,sqrt2)
        z_norm = torch.mul(div,sqrt1)
        return z_norm

        
class AWGN(nn.Module):
    def __init__(self,c,h,w,device):
        super(AWGN,self).__init__()
        self.c = c
        self.h = h
        self.w = w
        self.device = device
        
    # def forward(self,z_norm,snr_db):
    #     # 计算信噪比（SNR）对应的噪声方差
    #     snr = 10**(snr_db/10.0)      
    #     xpower = torch.sum(torch.abs(z_norm)**2,axis=-1,keepdims=True)/z_norm.shape[-1] # type: ignore
    #     npower = xpower / (2*snr)
    #     # 生成均值为0、标准差为npower的复数高斯噪声，并加到x上
    #     randn1 = torch.randn(z_norm.shape).to(self.device)
    #     randn2 = torch.randn(z_norm.shape).to(self.device)
    #     noise_real = randn1 * torch.sqrt(npower)
    #     noise_imag = randn2 * torch.sqrt(npower)
    #     noise = noise_real + 1j*noise_imag
    #     z_noised=z_norm + noise
    #     real_tensor = torch.real(z_noised)
    #     imag_tensor = torch.imag(z_noised)
    #     z_noised = torch.stack([real_tensor,imag_tensor],dim=1)
    #     z_noised = z_noised.view(-1,self.c,self.h,self.w)
    #     return z_noised
    
    def forward(self,z_norm,snr_db):
        snr_db = snr_db[0]
        # batch_size = z_norm.size(0)/self.c/self.h/self.w*2
        # z_norm = z_norm.view(int(batch_size), -1)
        # 计算信噪比的标准差
        noise_stddev = torch.sqrt(10 ** (-snr_db / 10.0))
        
        # 生成噪声
        noise_real = torch.randn_like(z_norm.real)
        noise_imag = torch.randn_like(z_norm.imag)
        # noise_stddev = noise_stddev.view(noise_stddev.size(0), -1)
        awgn = torch.complex(noise_real, noise_imag)

        # 将噪声标准差应用于噪声
        awgn *= noise_stddev.to(self.device)*1/torch.sqrt(torch.tensor(2))
        
        # 添加噪声到输入信号
        z_noised = z_norm + awgn
        real_tensor = torch.real(z_noised)
        imag_tensor = torch.imag(z_noised)
        z_noised = torch.stack([real_tensor,imag_tensor],dim=1)
        z_noised = z_noised.view(-1,self.c,self.h,self.w)
        return z_noised

class Fading(nn.Module):
    def __init__(self,c,h,w,device):
        super(Fading,self).__init__()
        self.c = c
        self.h = h
        self.w = w
        self.device = device

    def forward(self,z_norm,snr_db):
        # 计算信噪比（SNR）对应的噪声方差
        # snr = 10**(snr_db/10.0)     
        # xpower = torch.sum(torch.abs(z_norm)**2,axis=-1,keepdims=True)/z_norm.shape[-1] # type: ignore
        # npower = xpower / (2*snr) 
        # h = torch.complex(
        #     torch.randn(z_norm.shape ).to(self.device)*1 / np.sqrt(2),
        #     torch.randn(z_norm.shape).to(self.device)*1 / np.sqrt(2),
        # )

        # # additive white gaussian noise
        # awgn = torch.complex(
        #     torch.randn(z_norm.shape ).to(self.device)*1 / np.sqrt(2),
        #     torch.randn(z_norm.shape).to(self.device)*1 / np.sqrt(2),
        # )
        # z_noised = h * z_norm + torch.sqrt(npower) * awgn
        # z_noised = z_noised/h
        # real_tensor = torch.real(z_noised)
        # imag_tensor = torch.imag(z_noised)
        # z_noised = torch.stack([real_tensor,imag_tensor],dim=1)
        # z_noised = z_noised.view(-1,self.c,self.h,self.w)
        # def rayleigh_noise_layer(self, input_layer, std, name=None):
        input_layer = z_norm
        std = torch.sqrt(1.0 / (2 * 10 ** (snr_db / 10)))
        noise_real = torch.normal(mean=0.0, std=std[0], size=np.shape(input_layer))
        noise_imag = torch.normal(mean=0.0, std=std[0], size=np.shape(input_layer))
        noise = noise_real + 1j * noise_imag
        h = torch.sqrt(torch.normal(mean=0.0, std=1, size=np.shape(input_layer)) ** 2
                       + torch.normal(mean=0.0, std=1, size=np.shape(input_layer)) ** 2) / np.sqrt(2)
        
        noise = noise.to(input_layer.get_device())
        h = h.to(input_layer.get_device())
        z_noised = input_layer * h + noise
        real_tensor = torch.real(z_noised)
        imag_tensor = torch.imag(z_noised)
        z_noised = torch.stack([real_tensor,imag_tensor],dim=1)
        # z_noised = z_noised.view(-1,self.c,self.h,self.w)
        return z_noised.view(-1,self.c,self.h,self.w)
        # return z_noised
        


class AE_ori(nn.Module):
    def __init__(self,c,h,w,P,device):
        super(AE_ori,self).__init__()
        self.device = device
        self.multiple_snr = args.multiple_snr.split(",")
        self.Encoder = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=2,padding=2),
            nn.PReLU(num_parameters=16),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=2,padding=2),
            nn.PReLU(num_parameters=32),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.PReLU(num_parameters=32),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.PReLU(num_parameters=32),
            nn.Conv2d(in_channels=32,out_channels=c,kernel_size=5,stride=1,padding=2),
            nn.PReLU(num_parameters=c),
            
        )
        self.NormalizationLayer = NormalizationLayerE_complex(P)
        self.AWGN = AWGN(c,h,w,self.device)
        self.Fading = Fading(c,h,w,self.device)
        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=c,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.PReLU(num_parameters=32),
            nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.PReLU(num_parameters=32),
            nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.PReLU(num_parameters=32),
            nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=5,stride=2,padding=2,output_padding=1),
            nn.PReLU(num_parameters=16),
            nn.ConvTranspose2d(in_channels=16,out_channels=3,kernel_size=5,stride=2,padding=2,output_padding=1),
            nn.Sigmoid(),
            
        )

    def forward(self,x,snr_db,type='awgn'):
        z_encoded = self.Encoder(x)
        z_norm = self.NormalizationLayer(z_encoded)
        # KL divergence
        # KL = self.KL_log_uniform(10 ** (-snr_db / 20.0),torch.abs(z_norm))
        if type!='fading':
            z_noised = self.AWGN(z_norm,snr_db)
        else:
            z_noised = self.Fading(z_norm,snr_db)
        z_decoded = self.Decoder(z_noised)
        return z_decoded


class AE(nn.Module):
    def __init__(self,c,h,w,P,device):
        super(AE,self).__init__()
        self.device = device
        self.Encoder = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=2,padding=2),
            GDN(ch=16, device=device),
            nn.PReLU(num_parameters=16),
            AFModule(input_channels=16),

            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=2,padding=2),
            GDN(ch=32, device=device),
            nn.PReLU(num_parameters=32),
            AFModule(input_channels=32),

            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2),
            GDN(ch=32, device=device),
            nn.PReLU(num_parameters=32),
            AFModule(input_channels=32),

            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2),
            GDN(ch=32, device=device),
            nn.PReLU(num_parameters=32),
            AFModule(input_channels=32),

            nn.Conv2d(in_channels=32,out_channels=c,kernel_size=5,stride=1,padding=2),
            GDN(ch=c, device=device),
            # nn.PReLU(num_parameters=c),
            
        )
        self.NormalizationLayer = NormalizationLayerE_complex(P)
        self.AWGN = AWGN(c,h,w,self.device)
        self.Fading = Fading(c,h,w,self.device)
        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=c,out_channels=32,kernel_size=5,stride=1,padding=2),
            IGDN(ch=32, device=device),
            nn.PReLU(num_parameters=32),
            AFModule(input_channels=32),

            nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2),
            IGDN(ch=32, device=device),
            nn.PReLU(num_parameters=32),
            AFModule(input_channels=32),

            nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2),
            IGDN(ch=32, device=device),
            nn.PReLU(num_parameters=32),
            AFModule(input_channels=32),

            nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=5,stride=2,padding=2,output_padding=1),
            IGDN(ch=16, device=device),
            nn.PReLU(num_parameters=16),
            AFModule(input_channels=16),

            nn.ConvTranspose2d(in_channels=16,out_channels=3,kernel_size=5,stride=2,padding=2,output_padding=1),
            IGDN(ch=3, device=device),
            nn.Sigmoid(),
            
        )

    def forward_layer(self, layer, x, snr=None):
        if isinstance(layer, AFModule):
            return layer(x, snr)
        return layer(x)
    
    def forward(self,x,snr_db,type='awgn'):
        snr_db = snr_db.to(self.device)
        for layer in self.Encoder:
            x = self.forward_layer(layer, x, snr_db)
        z_norm = self.NormalizationLayer(x)
        # KL divergence
        # KL = self.KL_log_uniform(10 ** (-snr_db / 20.0),torch.abs(z_norm))
        if type!='fading':
            z_noised = self.AWGN(z_norm,snr_db)
        else:
            z_noised = self.Fading(z_norm,snr_db)
        for layer in self.Decoder:
            z_noised = self.forward_layer(layer, z_noised, snr_db)
        # z_encoded = self.Encoder(x)
        # z_norm = self.NormalizationLayer(z_encoded)
        # # KL divergence
        # # KL = self.KL_log_uniform(10 ** (-snr_db / 20.0),torch.abs(z_norm))
        # if type!='fading':
        #     z_noised = self.AWGN(z_norm,snr_db)
        # else:
        #     z_noised = self.Fading(z_norm,snr_db)
        # z_noised = self.Decoder(z_noised)

        return z_noised
    
   

class AE_256(nn.Module):
    def __init__(self,c,h,w,P,device):
        super(AE_256,self).__init__()
        self.device = device
        self.Encoder = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=256,kernel_size=9,stride=2,padding=2),
            GDN(ch=256, device=device),
            nn.PReLU(num_parameters=256),
            AFModule(input_channels=256),

            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=5,stride=2,padding=2),
            GDN(ch=256, device=device),
            nn.PReLU(num_parameters=256),
            AFModule(input_channels=256),

            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=5,stride=1,padding=2),
            GDN(ch=256, device=device),
            nn.PReLU(num_parameters=256),
            AFModule(input_channels=256),

            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=5,stride=1,padding=2),
            GDN(ch=256, device=device),
            nn.PReLU(num_parameters=256),
            AFModule(input_channels=256),

            nn.Conv2d(in_channels=256,out_channels=c,kernel_size=5,stride=1,padding=2),
            GDN(ch=c, device=device),
            # nn.PReLU(num_parameters=c),
            
        )
        self.NormalizationLayer = NormalizationLayerE_complex(P)
        self.AWGN = AWGN(c,h,w,self.device)
        self.Fading = Fading(c,h,w,self.device)
        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=c,out_channels=256,kernel_size=5,stride=1,padding=2),
            IGDN(ch=256, device=device),
            nn.PReLU(num_parameters=256),
            AFModule(input_channels=256),

            nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=5,stride=1,padding=2),
            IGDN(ch=256, device=device),
            nn.PReLU(num_parameters=256),
            AFModule(input_channels=256),

            nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=5,stride=1,padding=2),
            IGDN(ch=256, device=device),
            nn.PReLU(num_parameters=256),
            AFModule(input_channels=256),

            nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=5,stride=2,padding=2,output_padding=1),
            IGDN(ch=256, device=device),
            nn.PReLU(num_parameters=256),
            AFModule(input_channels=256),

            nn.ConvTranspose2d(in_channels=256,out_channels=3,kernel_size=9,stride=2,padding=2,output_padding=1),
            IGDN(ch=3, device=device),
            nn.Sigmoid(),
            
        )

    def forward_layer(self, layer, x, snr=None):
        if isinstance(layer, AFModule):
            return layer(x, snr)
        return layer(x)
    
    def forward(self,x,snr_db,type='awgn'):
        snr_db = snr_db.to(self.device)
        for layer in self.Encoder:
            x = self.forward_layer(layer, x, snr_db)
        z_norm = self.NormalizationLayer(x)
        # KL divergence
        # KL = self.KL_log_uniform(10 ** (-snr_db / 20.0),torch.abs(z_norm))
        if 'awgn' in type:
            z_noised = self.AWGN(z_norm,snr_db)
        elif 'fading' in type:
            z_noised = self.Fading(z_norm,snr_db)
        for layer in self.Decoder:
            z_noised = self.forward_layer(layer, z_noised, snr_db)

        return z_noised
    
   