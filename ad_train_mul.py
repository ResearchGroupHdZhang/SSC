import torch.optim as optim
from net.network_adjscc import WITT
import copy

from data.datasets_new import get_loader_multi
from utils import *
torch.backends.cudnn.benchmark = True
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
from datetime import datetime
import torch.nn as nn
import argparse
from loss.distortion import *
import time
import torchvision
from torch.utils.data import DataLoader
from src import utils
from src.prune import Pruning
from src.task import Task
from tqdm import tqdm

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CIFAR10WithSNR_multi(torch.utils.data.Dataset):
    def __init__(self, dataset, snr_values1,snr_values2):
        self.dataset = dataset
        self.snr_values1 = snr_values1
        self.snr_values2 = snr_values2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        snr1 = self.snr_values1[idx]
        snr2 = self.snr_values2[idx]
        return img, torch.tensor(snr1),torch.tensor(snr2)


def load_masks(masks_dir):
    # masks_path = [
    #     os.path.join(masks_dir, f)
    #     for f in os.listdir(masks_dir)
    #     if not f.startswith("init")
    # ]
    # masks_path = list(
    #     sorted(
    #         filter(lambda f: os.path.isfile(f), masks_path),
    #         key=lambda s: int(os.path.basename(s).split("_")[0]),
    #     )
    # )
    # masks_path = ['/home/zhoujh/code/saved_models/7dB_c16/sqrt_fading_final0.1_epoch300_1_79.43_noavgre25.3359432220459.th','/home/zhoujh/code/saved_models/17dB_c16/sqrt_fading_final0.1_epoch300_1_79.43_noavgre30.311365127563477.th']61.81766891479492
    # masks_path = ['/home/zhoujh/code/saved_models/7dB_c16/sqrt_fading_final0.8_epoch300_3_93.52_noavgre25.48944091796875.th','/home/zhoujh/code/saved_models/17dB_c16/sqrt_fading_final0.1_epoch300_2_63.1_noavgre30.455097198486328.th']#62.21249008178711
    # masks_path = ['/home/zhoujh/code/saved_models/7dB_c16/sqrt_fading_final0.8_epoch300_3_93.52_noavgre25.48944091796875.th','/home/zhoujh/code/saved_models/17dB_c16/sqrt_fading_final0.1_epoch300_4_39.81_noavgre30.080602645874023.th']#61.81766891479492
    # masks_path = ["/home/zhoujh/WITT-main/saved_models/div_0-10dB_C96_final_ratio0.8/<class 'type'>_final0.8_epoch500_7_85.54_noavgre0.th","/home/zhoujh/WITT-main/saved_models/div_10-20dB_C96_final_ratio0.8/<class 'type'>_final0.8_epoch500_7_85.54_noavgre0.th"]
    # masks_path = ["history/div_lre-5_lowdB_C96_final_ratio0.6/masks/<class 'type'>_final0.6_epoch19_4_81.52_noavgre31.83665691000043.th","/home/zhoujh/code/WITT-main/history/div_lre-5_highdB_C96_final_ratio0.6/masks/<class 'type'>_final0.6_epoch12_5_77.46_noavgre32.047255826718875.th"]
    # masks_path = ["/home/zhoujh/code/WITT-main/history/div_lre-5_lowdB_C96_final_ratio0.6/masks/<class 'type'>_final0.6_epoch18_10_60.0_noavgre32.24449174331897.th","/home/zhoujh/code/WITT-main/history/div_lre-5_highdB_C96_final_ratio0.6/masks/<class 'type'>_final0.6_epoch9_9_63.14_noavgre32.20187517368433.th"]
    # masks_path = ["/home/zhoujh/WITT-main/history/CIFAR10_awgn_lre-4_encoder,decoder_lowdB_C96_final_ratio0.6/masks/<class 'type'>_final0.6_epoch992_1_95.02_noavgre28.386200300852455.th","/home/zhoujh/WITT-main/history/CIFAR10_awgn_lre-4_encoder,decoder_highdB_C96_final_ratio0.6/masks/<class 'type'>_final0.6_epoch935_1_95.02_noavgre34.16810569763184.th"]
    # masks_path = ["/home/zhoujh/WITT-main/history/CIFAR10_rayleigh_lre-4_encoder,decoder_lowdB_C96_final_ratio0.6/masks/<class 'type'>_final0.6_epoch937_1_95.02_noavgre25.82018270492554.th","/home/zhoujh/WITT-main/history/CIFAR10_rayleigh_lre-4_encoder,decoder_highdB_C96_final_ratio0.6/masks/<class 'type'>_final0.6_epoch995_1_95.02_noavgre28.859355322519935.th"]
    masks_path = [
                  "/home/zhoujh/WITT-main/history/pre_CIFAR10_awgn_lre-4_encoder,decoder_highdB_C96_final_ratio0.6/masks/<class 'type'>_final0.6_epoch596_1_95.02_noavgre34.62233619689942.th",
                  "/home/zhoujh/WITT-main/history/pre_CIFAR10_awgn_lre-4_encoder,decoder_lowdB_C96_final_ratio0.6/masks/<class 'type'>_final0.6_epoch297_2_90.29_noavgre28.417069403330483.th",
                  ]

    # masks_path = ["/home/zhoujh/WITT-main/history/pre_CIFAR10_rayleigh_lre-4_encoder,decoder_lowdB_C96_final_ratio0.6/masks/<class 'type'>_final0.6_epoch596_1_95.02_noavgre25.901262474060058.th","/home/zhoujh/WITT-main/history/pre_CIFAR10_rayleigh_lre-4_encoder,decoder_highdB_C96_final_ratio0.6/masks/<class 'type'>_final0.6_epoch364_1_95.02_noavgre29.34442148208618.th"]

    masks = []
    logger.info("loading masks")
    for path in masks_path:
        dump = torch.load(path, "cpu")
        assert "mask" in dump and "pruning_time" in dump
        logger.info(
            "loading pruning_time {}, mask in {}".format(dump["pruning_time"], path)
        )
        masks.append(dump["mask"])

    # sanity check
    assert len(masks) == len(masks_path)
    for mi in masks:
        for name, m in mi.items():
            assert isinstance(m, torch.Tensor)
            mi[name] = m.bool()
    return masks
class MTL_Masker:
    def __init__(self, model, masks):
        self.model = model
        self.masks = masks
        self.weights = []
        if self.masks is None:
            mask = {}
            for name, param in self.model.named_parameters():
                m = torch.zeros_like(param.data).bool()
                mask[name] = m
            self.masks = mask
        logger.info("has masks %d, %s", len(self.masks), type(self.masks))

    def to(self, device):
        # logger.info(type(self.model), type(self.masks), device)
        logger.info("model to %s", device)
        self.model.to(device)
        if self.masks is None:
            return
        if isinstance(self.masks, dict):
            masks = [self.masks]
        else:
            masks = self.masks
        for i, mask in enumerate(masks):
            logger.info("mask {} to {}".format(i, device))
            for name, m in mask.items():
                mask[name] = m.to(device)

    def before_forward(self, task_id):
        # backup weights
        self.weights.append(copy.deepcopy(self.model.state_dict()))
        # apply mask to param
        self.apply_mask(task_id)

    def after_forward(self, task_id):
        # resume weights
        weights = self.weights.pop()
        model_dict = self.model.state_dict()
        checkpoint_dict = {k: v for k, v in weights.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(checkpoint_dict)
        self.model.load_state_dict(model_dict, strict=False)
        # apply mask to grad
        self.mask_grad(task_id)

    def apply_mask(self, task_id):
        if isinstance(self.masks, dict):
            mask = self.masks
        else:
            mask = self.masks[task_id]
        for name, param in self.model.named_parameters():
            if name in mask:
                param.data.masked_fill_(mask[name], 0.0)

    def mask_grad(self, task_id):
        # zero-out all the gradients corresponding to the pruned connections
        if isinstance(self.masks, dict):
            mask = self.masks
        else:
            mask = self.masks[task_id]
        for name, p in self.model.named_parameters():
            if name in mask and p.grad is not None:
                p.grad.data.masked_fill_(mask[name], 0.0)


parser = argparse.ArgumentParser(description='WITT')
# fmt: off

parser.add_argument("--masks_path", type=str, default=None, help='the task specific mask paths')
parser.add_argument("--tasks", type=str, default=None, help='the task ids for MTL, default using all tasks')
parser.add_argument("--trainer", type=str, choices=['re-seq-label', 'seq-label'], default='seq-label', help='the trainer type')
# fmt: on
parser.add_argument('--training', action='store_true',
                    help='training or testing')
parser.add_argument('--save_dir', type=str, default='/home/zhoujh/code/WITT-main/saved_models')
parser.add_argument('--init_weights', type=str, default='/home/zhoujh/code/WITT-main/saved_models/0-10dB_C96_final_ratio0.8/init_weights.th', help='init weights(checkpoints) for training')
parser.add_argument("--final_rate", dest='final_rate', type=float, default=0.6, help='percent of params to remain not to pruning')
parser.add_argument("--pruning_iter", dest='pruning_iter', type=int, default=10, help='max times to pruning')
parser.add_argument('--init_masks', dest='init_masks', type=str, default=None, help='initial masks for late reseting pruning')
parser.add_argument('--need_cut', default='encoder,decoder', type=str, dest='need_cut', help='parameters names that not cut')

parser.add_argument("--task_id", dest='task_id', type=int, default=0, help='the task to use')
parser.add_argument('--trainset', type=str, default='CIFAR10',
                    choices=['CIFAR10', 'DIV2K','IMG'],
                    help='train dataset name')
parser.add_argument('--testset', type=str, default='kodak',
                    choices=['kodak', 'CLIC21'],
                    help='specify the testset for HR models')
parser.add_argument('--distortion-metric', type=str, default='MSE',
                    choices=['MSE', 'MS-SSIM'],
                    help='evaluation metrics')
parser.add_argument('--model', type=str, default='WITT',
                    choices=['WITT', 'WITT_W/O'],
                    help='WITT model or WITT without channel ModNet')
parser.add_argument('--channel-type', type=str, default='awgn',
                    choices=['awgn', 'rayleigh'],
                    help='wireless channel model, awgn or rayleigh')
parser.add_argument('--C', type=int, default=16,
                    help='bottleneck dimension')
parser.add_argument('--multiple-snr', type=str, default='1,4,7,10,13',
                    help='random or fixed snr')
parser.add_argument('--mission', type=str, default='multi',
                    help='which mask will be made')
parser.add_argument('--cuda', type=int, default=0,
                    help='which card will be used')
args = parser.parse_args()
# 生成固定的 SNR 值
def generate_snr_values(length,low,high):
    random.seed(42)  # 固定随机种子以保证生成相同的 SNR 值
    return [random.uniform(low, high) for _ in range(length)]
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.cuda}"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mission = args.mission
epochs = 1280
task_name = f'{args.channel_type}_{args.trainset}_lre-4_epochs_{args.need_cut}_{epochs}_{mission}dB_C96_final_ratio{args.final_rate}'
class config():
    seed = 1024
    pass_channel = True
    CUDA = True
    device = torch.device("cuda:0")
    norm = False
    # logger
    print_step = 500
    plot_step = 10000
    filename = task_name
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    # masks = workdir + '/masks'
    logger = None
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(models, exist_ok=True)
    # os.makedirs(masks, exist_ok=True)
    # training details
    normalize = False
    learning_rate = 0.0001
    tot_epoch = epochs

    if args.trainset == 'CIFAR10':
        save_model_freq = 100
        image_dims = (3, 32, 32)
        train_data_dir = "/home/zhoujh/ADJSCC-main/sparse_sharing/dataset"
        test_data_dir = "/home/zhoujh/ADJSCC-main/sparse_sharing/dataset"
        batch_size = 128
        downsample = 2
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
    elif args.trainset == 'DIV2K':
        save_model_freq = 100
        image_dims = (3, 256, 256)
        train_data_dir = ["/home/zhoujh/code/dataset/div_train/DIV2K_train_HR/"]
        if args.testset == 'kodak':
            test_data_dir = ["/home/zhoujh/code/dataset/dataset_kodak/Kodak24/"]
        elif args.testset == 'CLIC21':
            test_data_dir = ["/media/Dataset/CLIC21/"]
        batch_size = 8
        downsample = 4
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
            C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4],
            C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
    elif args.trainset == 'IMG':
        save_model_freq = 5
        image_dims = (3, 128, 128)
        train_data_dir = ["/home/zhoujh/code/train_out/train"]
        if args.testset == 'kodak':
            test_data_dir = ["/home/zhoujh/code/dataset/dataset_kodak/Kodak24/"]
        elif args.testset == 'CLIC21':
            test_data_dir = ["/media/Dataset/CLIC21/"]
        batch_size = 16
        downsample = 4
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
            C=args.C, window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4],
            C=args.C, window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )

if args.trainset == 'CIFAR10':
    CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
else:
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()

def load_weights(model_path):
    pretrained = torch.load(model_path)
    net.load_state_dict(pretrained, strict=True)
    del pretrained


def train_one_epoch(args):
    net.train()
    elapsed1, losses1, psnrs1, msssims1, cbrs1, snrs1 = [AverageMeter() for _ in range(6)]
    elapsed2, losses2, psnrs2, msssims2, cbrs2, snrs2 = [AverageMeter() for _ in range(6)]
    metrics1 = [elapsed1, losses1, psnrs1, msssims1, cbrs1, snrs1]
    metrics2 = [elapsed2, losses2, psnrs2, msssims2, cbrs2, snrs2]
    global global_step
    if args.trainset == 'CIFAR10':
        loop = tqdm((train_data_loader),total=len(train_data_loader))
        for (input,snr1,snr2) in loop:
            global_step += 1
            input = input.to(device)
            for task in task_lst:
                masker.before_forward(task_id=task.task_id)
                start_time = time.time()
                
                if task.value == 1:
                    snr1_batch = torch.tensor(snr1).expand(len(input)).unsqueeze(1).cuda()
                    recon_image, CBR, SNR,  loss_G = net(input,snr1_batch)
                    # recon_image, CBR, SNR, mse, loss_G = net(input,snr)
                    loss = loss_G
                    optimizer.zero_grad()
                    loss.backward()
                    masker.after_forward(task.task_id)

                    optimizer.step()
                    elapsed1.update(time.time() - start_time)
                    losses1.update(loss.item())
                    cbrs1.update(CBR)
                    snrs1.update(SNR[0][0])
                    # if mse.item() > 0:
                    #     psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                    #     psnrs1.update(psnr.item())
                    #     msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                    #     msssims1.update(msssim)
                    # else:
                    #     psnrs1.update(100)
                    #     msssims1.update(100)
                    loop.set_description(f'Epoch [{epoch}/{config.tot_epoch}]')
                    loop.set_postfix(loss = loss.item())
                    if (global_step % config.print_step) == 0:
                        process = (global_step % train_data_loader.__len__()) / (train_data_loader.__len__()) * 100.0
                        log = (' | '.join([
                            f'task {task.task_id}',
                            f'Epoch {epoch}',
                            f'Step [{global_step % train_data_loader.__len__()}/{train_data_loader.__len__()}={process:.2f}%]',
                            f'Time {elapsed1.val:.3f}',
                            f'Loss {losses1.val:.3f} ({losses1.avg:.3f})',
                            f'CBR {cbrs1.val:.4f} ({cbrs1.avg:.4f})',
                            f'SNR {snrs1.val:.1f} ({snrs1.avg:.1f})',
                            # f'PSNR {psnrs1.val:.3f} ({psnrs1.avg:.3f})',
                            # f'MSSSIM {msssims1.val:.3f} ({msssims1.avg:.3f})',
                            f'Lr {cur_lr}',
                        ]))
                        # PSNR = psnrs1.avg
                        logger.info(log)
                        for i in metrics1:
                            i.clear()
                else: 
                    snr2_batch = torch.tensor(snr2).expand(len(input)).unsqueeze(1).cuda()
                    recon_image, CBR, SNR,  loss_G = net(input,snr2_batch)
                    loss = loss_G
                    optimizer.zero_grad()
                    loss.backward()
                    masker.after_forward(task.task_id)

                    optimizer.step()
                    elapsed2.update(time.time() - start_time)
                    losses2.update(loss.item())
                    cbrs2.update(CBR)
                    snrs2.update(SNR[0][0])
                    # if mse.item() > 0:
                    #     psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                    #     psnrs2.update(psnr.item())
                    #     msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                    #     msssims2.update(msssim)
                    # else:
                    #     psnrs2.update(100)
                    #     msssims2.update(100)
                    loop.set_description(f'Epoch [{epoch}/{config.tot_epoch}]')
                    loop.set_postfix(loss = loss.item())
                    if (global_step % config.print_step) == 0:
                        process = (global_step % train_data_loader.__len__()) / (train_data_loader.__len__()) * 100.0
                        log = (' | '.join([
                            f'task {task.task_id}',
                            f'Epoch {epoch}',
                            f'Step [{global_step % train_data_loader.__len__()}/{train_data_loader.__len__()}={process:.2f}%]',
                            f'Time {elapsed2.val:.3f}',
                            f'Loss {losses2.val:.3f} ({losses2.avg:.3f})',
                            f'CBR {cbrs2.val:.4f} ({cbrs2.avg:.4f})',
                            f'SNR {snrs2.val:.1f} ({snrs2.avg:.1f})',
                            # f'PSNR {psnrs2.val:.3f} ({psnrs2.avg:.3f})',
                            # f'MSSSIM {msssims2.val:.3f} ({msssims2.avg:.3f})',
                            f'Lr {cur_lr}',
                        ]))
                        # PSNR = psnrs2.avg
                        logger.info(log)
                        for i in metrics2:
                            i.clear()
                
    else:
        loop = tqdm((train_data_loader),total=len(train_data_loader))
        for (input,snr1,snr2)  in loop:
            start_time = time.time()
            global_step += 1
            input = input.to(device)
            for task in task_lst:
                masker.before_forward(task_id=task.task_id)
                
                if task.value == 1:
                    snr1_batch = torch.tensor(snr1).expand(len(input))
                    recon_image, CBR, SNR, mse, loss_G = net(input,snr1_batch)
                    loss = loss_G
                    optimizer.zero_grad()
                    loss.backward()
                    masker.after_forward(task.task_id)
                    optimizer.step()
                    elapsed1.update(time.time() - start_time)
                    losses1.update(loss.item())
                    cbrs1.update(CBR)
                    snrs1.update(SNR[0])
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs1.update(psnr.item())
                        msssim = 1 - loss_G
                        msssims1.update(msssim)

                    else:
                        psnrs1.update(100)
                        msssims1.update(100)
                    loop.set_description(f'Epoch [{epoch}/{config.tot_epoch}]')
                    loop.set_postfix(loss = loss.item())
                    if (global_step % config.print_step) == 0:
                        process = (global_step % train_data_loader.__len__()) / (train_data_loader.__len__()) * 100.0
                        log = (' | '.join([
                            f'task {task.task_id}',
                            f'Epoch {epoch}',
                            f'Step [{global_step % train_data_loader.__len__()}/{train_data_loader.__len__()}={process:.2f}%]',
                            f'Time {elapsed1.val:.3f}',
                            f'Loss {losses1.val:.3f} ({losses1.avg:.3f})',
                            f'CBR {cbrs1.val:.4f} ({cbrs1.avg:.4f})',
                            f'SNR {snrs1.val:.1f} ({snrs1.avg:.1f})',
                            f'PSNR {psnrs1.val:.3f} ({psnrs1.avg:.3f})',
                            f'MSSSIM {msssims1.val:.3f} ({msssims1.avg:.3f})',
                            f'Lr {cur_lr}',
                        ]))
                        PSNR1 = psnrs1.avg
                        logger.info(log)
                        for i in metrics1:
                            i.clear()
                else:
                    snr2_batch = torch.tensor(snr2).expand(len(input))
                    recon_image, CBR, SNR, mse, loss_G = net(input,snr2_batch)
                    loss = loss_G
                    optimizer.zero_grad()
                    loss.backward()
                    masker.after_forward(task.task_id)
                    optimizer.step()
                    elapsed2.update(time.time() - start_time)
                    losses2.update(loss.item())
                    cbrs2.update(CBR)
                    snrs2.update(SNR[0])
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs2.update(psnr.item())
                        msssim = 1 - loss_G
                        msssims2.update(msssim)

                    else:
                        psnrs2.update(100)
                        msssims2.update(100)
                    loop.set_description(f'Epoch [{epoch}/{config.tot_epoch}]')
                    loop.set_postfix(loss = loss.item())
                    if (global_step % config.print_step) == 0:
                        process = (global_step % train_data_loader.__len__()) / (train_data_loader.__len__()) * 100.0
                        log = (' | '.join([
                            f'task {task.task_id}',
                            f'Epoch {epoch}',
                            f'Step [{global_step % train_data_loader.__len__()}/{train_data_loader.__len__()}={process:.2f}%]',
                            f'Time {elapsed2.val:.3f}',
                            f'Loss {losses2.val:.3f} ({losses2.avg:.3f})',
                            f'CBR {cbrs2.val:.4f} ({cbrs2.avg:.4f})',
                            f'SNR {snrs2.val:.1f} ({snrs2.avg:.1f})',
                            f'PSNR {psnrs2.val:.3f} ({psnrs2.avg:.3f})',
                            f'MSSSIM {msssims2.val:.3f} ({msssims2.avg:.3f})',
                            f'Lr {cur_lr}',
                        ]))
                        
                        logger.info(log)
                        PSNR2 = psnrs2.avg
                        for i in metrics2:
                            i.clear()

    # for i in metrics:
    #     i.clear()
    # return PSNR1+PSNR2

def test():
    config.isTrain = False
    net.eval()
    elapsed1, psnrs1, msssims1, snrs1, cbrs1 = [AverageMeter() for _ in range(5)]
    elapsed2, psnrs2, msssims2, snrs2, cbrs2 = [AverageMeter() for _ in range(5)]
    metrics1 = [elapsed1, psnrs1, msssims1, snrs1, cbrs1]
    metrics2 = [elapsed2, psnrs2, msssims2, snrs2, cbrs2]
    # multiple_snr = args.multiple_snr.split(",")
    # for i in range(len(multiple_snr)):
    #     multiple_snr[i] = int(multiple_snr[i])
    # results_snr = np.zeros(len(multiple_snr))
    # results_cbr = np.zeros(len(multiple_snr))
    # results_psnr = np.zeros(len(multiple_snr))
    # results_msssim = np.zeros(len(multiple_snr))
    snr_set=[0,2,4,6,8,10,12,14,16,18,20]
    results_snr1 = np.zeros(len(snr_set))
    results_cbr1 = np.zeros(len(snr_set))
    results_psnr1 = np.zeros(len(snr_set))
    results_msssim1 = np.zeros(len(snr_set))
    results_snr2 = np.zeros(len(snr_set))
    results_cbr2 = np.zeros(len(snr_set))
    results_psnr2 = np.zeros(len(snr_set))
    results_msssim2 = np.zeros(len(snr_set))
    for i,snr in  enumerate(snr_set):
        with torch.no_grad():
            if args.trainset == 'CIFAR10':
                for batch_idx, (input, snr1,snr2) in enumerate(test_data_loader):
                    start_time = time.time()
                    input = input.cuda()
                    snr_batch = torch.tensor(snr).expand(len(input)).unsqueeze(1).cuda()
                    for task in task_lst:
                        if task.value==1:
                            masker.before_forward(task.task_id)
                            # snr_batch = torch.tensor(snr).expand(len(input)).unsqueeze(1).cuda()
                            recon_image, CBR, SNR, mse, loss_G = net(input, snr_batch,False)
                            masker.after_forward(task.task_id)
                            elapsed1.update(time.time() - start_time)
                            cbrs1.update(CBR)
                            snrs1.update(SNR[0][0])
                            if mse.item() > 0:
                                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                                psnrs1.update(psnr.item())
                                msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                                msssims1.update(msssim)
                            else:
                                psnrs1.update(100)
                                msssims1.update(100)
                        else:
                            masker.before_forward(task.task_id)
                            
                            recon_image, CBR, SNR, mse, loss_G = net(input, snr_batch,False)
                            masker.after_forward(task.task_id)
                            elapsed2.update(time.time() - start_time)
                            cbrs2.update(CBR)
                            snrs2.update(SNR[0][0])
                            if mse.item() > 0:
                                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                                psnrs2.update(psnr.item())
                                msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                                msssims2.update(msssim)
                            else:
                                psnrs2.update(100)
                                msssims2.update(100)

                log1 = (' | '.join([
                    f'Time {elapsed1.val:.3f}',
                    f'CBR {cbrs1.val:.4f} ({cbrs1.avg:.4f})',
                    f'SNR {snrs1.val:.1f}',
                    f'PSNR {psnrs1.val:.3f} ({psnrs1.avg:.3f})',
                    f'MSSSIM {msssims1.val:.3f} ({msssims1.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                log2 = (' | '.join([
                    f'Time {elapsed2.val:.3f}',
                    f'CBR {cbrs2.val:.4f} ({cbrs2.avg:.4f})',
                    f'SNR {snrs2.val:.1f}',
                    f'PSNR {psnrs2.val:.3f} ({psnrs2.avg:.3f})',
                    f'MSSSIM {msssims2.val:.3f} ({msssims2.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                # logger.info(log1)
                # logger.info(log2)
                # print(log1)
                # print(log2)
            else:
                for batch_idx, (input, snr1,snr2) in enumerate(test_data_loader):
                    start_time = time.time()
                    input = input.cuda()
                    for task in task_lst:
                        if task.value==1:
                            masker.before_forward(task.task_id)
                            snr_batch = torch.full((len(input),), snr)
                            recon_image, CBR, SNR, mse, loss_G = net(input, snr_batch)
                            masker.after_forward(task.task_id)
                            elapsed1.update(time.time() - start_time)
                            cbrs1.update(CBR)
                            snrs1.update(snr)
                            if mse.item() > 0:
                                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                                psnrs1.update(psnr.item())
                                msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                                msssims1.update(msssim)
                            else:
                                psnrs1.update(100)
                                msssims1.update(100)
                        else:
                            masker.before_forward(task.task_id)
                            
                            recon_image, CBR, SNR, mse, loss_G = net(input, snr_batch)
                            masker.after_forward(task.task_id)
                            elapsed2.update(time.time() - start_time)
                            cbrs2.update(CBR)
                            snrs2.update(snr)
                            if mse.item() > 0:
                                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                                psnrs2.update(psnr.item())
                                msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                                msssims2.update(msssim)
                            else:
                                psnrs2.update(100)
                                msssims2.update(100)

                log1 = (' | '.join([
                f'Time {elapsed1.val:.3f}',
                f'CBR {cbrs1.val:.4f} ({cbrs1.avg:.4f})',
                f'SNR {snrs1.val:.1f}',
                f'PSNR {psnrs1.val:.3f} ({psnrs1.avg:.3f})',
                f'MSSSIM {msssims1.val:.3f} ({msssims1.avg:.3f})',
                f'Lr {cur_lr}',
                ]))
                log2 = (' | '.join([
                    f'Time {elapsed2.val:.3f}',
                    f'CBR {cbrs2.val:.4f} ({cbrs2.avg:.4f})',
                    f'SNR {snrs2.val:.1f}',
                    f'PSNR {psnrs2.val:.3f} ({psnrs2.avg:.3f})',
                    f'MSSSIM {msssims2.val:.3f} ({msssims2.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                # logger.info(log1)
                # logger.info(log2)
                # print(log1)
                # print(log2)
        results_snr1[i] = snrs1.avg
        results_cbr1[i] = cbrs1.avg
        results_psnr1[i] = psnrs1.avg
        results_msssim1[i] = msssims1.avg
        results_snr2[i] = snrs2.avg
        results_cbr2[i] = cbrs2.avg
        results_psnr2[i] = psnrs2.avg
        results_msssim2[i] = msssims2.avg
        for t in metrics1:
            t.clear()
        for t in metrics2:
            t.clear()

    # print("SNR: {}" .format(results_snr.tolist()))
    # print("CBR: {}".format(results_cbr.tolist()))
    # print("PSNR: {}" .format(results_psnr.tolist()))
    # print("MS-SSIM: {}".format(results_msssim.tolist()))
    logger.info("SNR: {}" .format(results_snr1.tolist()))
    logger.info("CBR: {}".format(results_cbr1.tolist()))
    logger.info("PSNR: {}" .format(results_psnr1.tolist()))
    logger.info("MS-SSIM: {}".format(results_msssim1.tolist()))
    logger.info("SNR: {}" .format(results_snr2.tolist()))
    logger.info("CBR: {}".format(results_cbr2.tolist()))
    logger.info("PSNR: {}" .format(results_psnr2.tolist()))
    logger.info("MS-SSIM: {}".format(results_msssim2.tolist()))
    logger.info("Finish Test!")

if __name__ == '__main__':
    seed_torch()
    
    # parser = utils.get_default_parser()

    

    args = parser.parse_args()

    # utils.init_prog(args)
    # task_name = "c256_0-20_div"
    logger = logger_configuration(config, save_log=True)
    logger.info(config.__dict__)

    logger.info(args)
    # torch.save(args, os.path.join(args.save_path, "args.th"))

    n_gpu = torch.cuda.device_count()
    print("# of gpu: {}".format(n_gpu))

    logger.info("========== Loading Datasets ==========")
    task_lst= [Task(task_id=0,task_name='0-10dB',value = 1),Task(task_id=1,task_name='10-20dB',value = 2)]
    if args.tasks is not None:
        args.tasks = list(map(int, map(lambda s: s.strip(), args.tasks.split(","))))
        logger.info("activate tasks %s", args.tasks)
    logger.info("# of Tasks: {}.".format(len(task_lst)))
    for task in task_lst:
        logger.info("Task {}: {}".format(task.task_id, task.task_name))
    # for task in task_lst:
    #     if args.debug:
    #         task.train_set = task.train_set[:200]
    #         task.dev_set = task.dev_set[:200]
    #         task.test_set = task.test_set[:3200]
    #         args.epochs = 3
    #     task.init_data_loader(args.batch_size)
    logger.info("done.")

    # model_descript = args.exp_name
    # print('====== Loading Word Embedding =======')

    logger.info("========== Preparing Model ==========")

    # n_class_per_task = []
    # for task in task_lst:
    #     n_class_per_task.append(len(vocabs[task.task_name]))
    # logger.info("n_class %s", n_class_per_task)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data = torchvision.datasets.CIFAR10(root='/home/zhoujh/ADJSCC-main/sparse_sharing/dataset',train=True,transform=torchvision.transforms.ToTensor(),download=True)
    # dev_data = task_db.dev_set
    test_data = torchvision.datasets.CIFAR10(root='/home/zhoujh/ADJSCC-main/sparse_sharing/dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)
    low1 = 0
    high1 = 10
    low2 = 10
    high2 = 20
    train_data_loader, test_data_loader = get_loader_multi(args, config,low1,high1,low2,high2)
    # logger.info("task name: {}, task id: {}".format(task_name, task_id))
    # logger.info(
    #     "train len {}, test len {}".format(
    #         len(train_data), len(test_data)
    #     )
    # )
    # logger = logger_configuration(config, save_log=True)
    # logger.info(config.__dict__)
    torch.manual_seed(seed=config.seed)
    net = WITT(args, config,device)
    logger.info("model: \n{}".format(net))
    model_path = '/home/zhoujh/WITT-main/history/awgn_CIFAR10_dynamicdB_C16_final_ratio0.8/models/awgn_CIFAR10_dynamicdB_C16_final_ratio0.8_awgn_CIFAR10_dynamicdB_C16_final_ratio0.8_EP885.model'
    load_weights(model_path)
    masks = load_masks(args.masks_path)

    # if args.init_weights is not None:
    #     utils.load_model(net, args.init_weights)
    # if args.masks_path is not None:
    #     utils.load_model(net, os.path.join(args.masks_path, "init_weights"))
    masker = MTL_Masker(net, masks)

    logger.info("Model parameters:")
    params = list(net.named_parameters())
    sum_param = 0
    for name, param in params:
        if param.requires_grad:
            logger.info("{}: {}".format(name, param.shape))
            sum_param += param.numel()
    logger.info("# Parameters: {}.".format(sum_param))
    masker.to("cuda:0" if torch.cuda.is_available() else "cpu")
    
    args.save_path = os.path.join(args.save_dir, task_name)
    os.makedirs(args.save_path, exist_ok=True)
    
    # net = net.to(device)
    need_cut_names = list(set([s.strip() for s in args.need_cut.split(",")]))
    prune_names = []
    for name, p in net.named_parameters():
        if not p.requires_grad or "bias" in name:
            continue
        for n in need_cut_names:
            if n in name:
                prune_names.append(name)
                break

    
    model_params = [{'params': net.parameters(), 'lr': 0.0001}]
    # train_data_loader, test_data_loader = get_loader(args, config)
    cur_lr = config.learning_rate
    optimizer = optim.Adam(model_params, lr=cur_lr)
    global_step = 0
    steps_epoch = global_step // train_data_loader.__len__()
    if False:
        logger.info("========== Training Model ==========")
        # base_params = filter(lambda p: p.requires_grad, net.parameters())
        # opt = utils.get_optim(args.optim, base_params)
        # logger.info(opt)
        for epoch in range(steps_epoch, config.tot_epoch):
            train_one_epoch(args)
            if (epoch + 1) % config.save_model_freq == 0:
                print('save')
                save_model(net, save_path=config.models + '/{}_EP{}.model'.format(config.filename, epoch + 1))
            res = test()
     
    else:
        logger.info("========== Evaluating Model ==========")
        model_path = "/home/zhoujh/WITT-main/history/awgn_CIFAR10_lre-4_epochs_encoder,decoder_1280_multidB_C96_final_ratio0.6/models/awgn_CIFAR10_lre-4_epochs_encoder,decoder_1280_multidB_C96_final_ratio0.6_EP700.model"
        load_weights(model_path)
        test()