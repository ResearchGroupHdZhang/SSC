import torch.optim as optim
from net.network_adjscc import WITT
from data.datasets_new import get_loader
from utils import *
torch.backends.cudnn.benchmark = True
import torch
from datetime import datetime
import torch.nn as nn
import argparse
from loss.distortion import *
import time
import torchvision
from torch.utils.data import DataLoader, Dataset
from src import utils
from src.prune import Pruning
from tqdm import tqdm

class CIFAR10WithBatchSNR(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]



parser = argparse.ArgumentParser(description='WITT')
parser.add_argument('--training', action='store_true',
                    help='training or testing')
parser.add_argument('--save_dir', type=str, default='/home/zhoujh/WITT-main/saved_models')
parser.add_argument('--init_weights', type=str, default='/home/zhoujh/WITT-main/saved_models/0-10dB_c16/init_weights.th', help='init weights(checkpoints) for training')
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
parser.add_argument('--multiple-snr', type=str, default='0,2,4,6,8,10,12,14,16,18,20',
                    help='random or fixed snr')
parser.add_argument('--mission', type=str, default='high',
                    help='which mask will be made')
parser.add_argument('--cuda', type=int, default=1,
                    help='which card will be used')
args = parser.parse_args()
# 生成固定的 SNR 值
def generate_snr_values(length,low,high):
    random.seed(42)  # 固定随机种子以保证生成相同的 SNR 值
    return [random.uniform(low, high) for _ in range(length)]
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.cuda}"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mission = args.mission
task_name = f'pre_{args.trainset}_{args.channel_type}_lre-4_{args.need_cut}_{mission}dB_C96_final_ratio{args.final_rate}'
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
    masks = workdir + '/masks'
    logger = None
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(models, exist_ok=True)
    os.makedirs(masks, exist_ok=True)
    # training details
    normalize = False
    learning_rate = 0.0001
    if args.channel_type !='awgn':
        tot_epoch = 1000
    else:
        tot_epoch = 1000

    if args.trainset == 'CIFAR10':
        save_model_freq = 5
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
    elapsed, losses, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(6)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs]
    global global_step
    if args.trainset == 'CIFAR10':
        loop = tqdm((train_data_loader),total=len(train_data_loader))
        for (input,snr) in loop:
            start_time = time.time()
            global_step += 1
            input = input.to(device)
            snr_batch = torch.tensor(snr).expand(len(input)).unsqueeze(1).cuda()
            recon_image, CBR, SNR, loss_G = net(input,snr_batch)
            loss = loss_G
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pruner.on_batch_end()
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            snrs.update(SNR[0][0])
            loop.set_description(f'Epoch [{epoch}/{config.tot_epoch}]')
            loop.set_postfix(loss = loss.item())
            # if mse.item() > 0:
            #     psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
            #     psnrs.update(psnr.item())
            #     msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
            #     msssims.update(msssim)
            # else:
            #     psnrs.update(100)
            #     msssims.update(100)

            if (global_step % config.print_step) == 0:
                process = (global_step % train_data_loader.__len__()) / (train_data_loader.__len__()) * 100.0
                log = (' | '.join([
                    f'Epoch {epoch}',
                    f'Step [{global_step % train_data_loader.__len__()}/{train_data_loader.__len__()}={process:.2f}%]',
                    f'Time {elapsed.val:.3f}',
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                    # f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    # f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                PSNR = psnrs.avg
                logger.info(log)
                for i in metrics:
                    i.clear()
    else:
        for batch_idx, (input,snr) in enumerate(train_data_loader):
            start_time = time.time()
            global_step += 1
            input = input.cuda()
            snr_batch = torch.tensor(snr).expand(len(input))
            recon_image, CBR, SNR,  loss_G = net(input,snr_batch)
            loss = loss_G
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pruner.on_batch_end()
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            snrs.update(SNR[0])
            # if mse.item() > 0:
            #     psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
            #     psnrs.update(psnr.item())
            #     msssim = 1 - loss_G
            #     msssims.update(msssim)

            # else:
            #     psnrs.update(100)
            #     msssims.update(100)

            if (global_step % config.print_step) == 0:
                process = (global_step % train_data_loader.__len__()) / (train_data_loader.__len__()) * 100.0
                log = (' | '.join([
                    f'Epoch {epoch}',
                    f'Step [{global_step % train_data_loader.__len__()}/{train_data_loader.__len__()}={process:.2f}%]',
                    f'Time {elapsed.val:.3f}',
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                    # f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    # f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                logger.info(log)
                for i in metrics:
                    i.clear()
    for i in metrics:
        i.clear()
    # return PSNR

def test():
    config.isTrain = False
    net.eval()
    elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs]
    multiple_snr = args.multiple_snr.split(",")
    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])
    results_snr = np.zeros(len(multiple_snr))
    results_cbr = np.zeros(len(multiple_snr))
    results_psnr = np.zeros(len(multiple_snr))
    results_msssim = np.zeros(len(multiple_snr))
    for i, SNR in enumerate(multiple_snr):
        with torch.no_grad():
            if args.trainset == 'CIFAR10':
                for batch_idx, (input, label) in enumerate(test_data_loader):
                    start_time = time.time()
                    input = input.cuda()
                    snr_batch = torch.tensor(SNR).expand(len(input)).unsqueeze(1).cuda()
                    recon_image, CBR, SNR_re, mse, loss_G = net(input,snr_batch,False)
                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR_re[0][0])
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                        msssims.update(msssim)
                    else:
                        psnrs.update(100)
                        msssims.update(100)

                    log = (' | '.join([
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Lr {cur_lr}',
                    ]))
                    # logger.info(log)
            elif args.trainset == 'DIV2K':
                for batch_idx, input in enumerate(test_data_loader):
                    start_time = time.time()
                    input = input.cuda()
                    recon_image, CBR, SNR, mse, loss_G = net(input,SNR)
                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR)
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                        msssims.update(msssim)
                    else:
                        psnrs.update(100)
                        msssims.update(100)

                    log = (' | '.join([
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Lr {cur_lr}',
                    ]))
                    logger.info(log)
            else:
                for batch_idx, (input,_) in enumerate(test_data_loader):
                    start_time = time.time()
                    input = input.cuda()
                    snr_batch = torch.tensor(SNR).expand(len(input))
                    recon_image, CBR, SNR, mse, loss_G = net(input,snr_batch)
                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR[0][0])
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                        msssims.update(msssim)
                    else:
                        psnrs.update(100)
                        msssims.update(100)

                    log = (' | '.join([
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Lr {cur_lr}',
                    ]))
                    # logger.info(log)
        results_snr[i] = snrs.avg
        results_cbr[i] = cbrs.avg
        results_psnr[i] = psnrs.avg
        results_msssim[i] = msssims.avg
        for t in metrics:
            t.clear()

    # print("SNR: {}" .format(results_snr.tolist()))
    # print("CBR: {}".format(results_cbr.tolist()))
    # print("PSNR: {}" .format(results_psnr.tolist()))
    # print("MS-SSIM: {}".format(results_msssim.tolist()))
    # print("Finish Test!")
    logger.info("SNR: {}" .format(results_snr.tolist()))
    logger.info("CBR: {}".format(results_cbr.tolist()))
    logger.info("PSNR: {}" .format(results_psnr.tolist()))
    logger.info("MS-SSIM: {}".format(results_msssim.tolist()))
    logger.info("Finish Test!")
    return np.mean(results_psnr)

def test_only():
    config.isTrain = False
    net.eval()
    elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs]
    # multiple_snr = args.multiple_snr.split(",")
    # for i in range(len(multiple_snr)):
    #     multiple_snr[i] = int(multiple_snr[i])
    # results_snr = np.zeros(len(multiple_snr))
    # results_cbr = np.zeros(len(multiple_snr))
    # results_psnr = np.zeros(len(multiple_snr))
    # results_msssim = np.zeros(len(multiple_snr))
    
    with torch.no_grad():
        if args.trainset == 'CIFAR10':
            for batch_idx, (input, snr) in enumerate(test_data_loader):
                start_time = time.time()
                input = input.cuda()
                snr_batch = torch.tensor(snr).expand(len(input))
                recon_image, CBR, SNR, mse, loss_G = net(input, snr_batch)
                elapsed.update(time.time() - start_time)
                cbrs.update(CBR)
                # snrs.update(snr)
                if mse.item() > 0:
                    psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                    psnrs.update(psnr.item())
                    msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                    msssims.update(msssim)
                else:
                    psnrs.update(100)
                    msssims.update(100)

            log = (' | '.join([
                f'Time {elapsed.val:.3f}',
                f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                # f'SNR {snrs.val:.1f}',
                f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                f'Lr {cur_lr}',
            ]))
            print(log)
        else:
            for batch_idx, (input,snr) in enumerate(test_data_loader):
                start_time = time.time()
                input = input.cuda()
                snr_batch = torch.tensor(snr).expand(len(input))
                recon_image, CBR, SNR, mse, loss_G = net(input, snr_batch)
                elapsed.update(time.time() - start_time)
                cbrs.update(CBR)
                snrs.update(SNR)
                if mse.item() > 0:
                    psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                    psnrs.update(psnr.item())
                    msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                    msssims.update(msssim)
                else:
                    psnrs.update(100)
                    msssims.update(100)

                log = (' | '.join([
                    f'Time {elapsed.val:.3f}',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    # f'SNR {snrs.val:.1f}',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                print(log)
                PSNR = psnrs.avg
        # results_snr[i] = snrs.avg
        # results_cbr[i] = cbrs.avg
        # results_psnr[i] = psnrs.avg
        # results_msssim[i] = msssims.avg
        for t in metrics:
            t.clear()
    
    # print("SNR: {}" .format(results_snr.tolist()))
    # print("CBR: {}".format(results_cbr.tolist()))
    # print("PSNR: {}" .format(results_psnr.tolist()))
    # print("MS-SSIM: {}".format(results_msssim.tolist()))
    print("Finish Test!")
    print(PSNR)
    return PSNR

if __name__ == '__main__':
    seed_torch()
    if mission=='low':
        low = 0
        high = 10
        args.multiple_snr = '0,2,4,6,8,10'
    else:
        low = 10
        high = 20
        args.multiple_snr = '10,12,14,16,18,20'
    
    # final_rate = 0.8
    # task_name = f'div_{low}-{high}dB_C96_final_ratio{args.final_rate}'
    logger = logger_configuration(config, save_log=True)
    logger.info(config.__dict__)
    logger.info(args)
    task_id = '1'
    logger.info(f'tot_epoch:{config.tot_epoch}')
    train_data_loader, test_data_loader = get_loader(args, config,low,high)
    logger.info("task name: {}, task id: {}".format(task_name, task_id))

    torch.manual_seed(seed=config.seed)
    net = WITT(args, config,device)
    logger.info("model: \n{}".format(net))
    # model_path = "/home/zhoujh/WITT-main/history/rayleigh_CIFAR10_dynamicdB_C16_final_ratio0.8/models/rayleigh_CIFAR10_dynamicdB_C16_final_ratio0.8_rayleigh_CIFAR10_dynamicdB_C16_final_ratio0.8_EP810.model"
    model_path = '/home/zhoujh/WITT-main/history/awgn_CIFAR10_dynamicdB_C16_final_ratio0.8/models/awgn_CIFAR10_dynamicdB_C16_final_ratio0.8_awgn_CIFAR10_dynamicdB_C16_final_ratio0.8_EP885.model'
    load_weights(model_path)
    
    
    args.save_path = os.path.join(args.save_dir, task_name)
    os.makedirs(args.save_path, exist_ok=True)
    if not os.path.exists(args.init_weights):
        logger.info('Saving init-weights to {}'.format(args.save_path))
        torch.save(
            net.cpu().state_dict(), os.path.join(args.save_path, "init_weights.th")
        )
    torch.save(args, os.path.join(args.save_path, "args.th"))
    if args.init_weights is not None:
        if os.path.exists(args.init_weights):
            print('taking init!')
            utils.load_model(net, args.init_weights)
    net = net.to(device)
    need_cut_names = list(set([s.strip() for s in args.need_cut.split(",")]))
    prune_names = []
    for name, p in net.named_parameters():
        if not p.requires_grad or "bias" in name:
            continue
        for n in need_cut_names:
            if n in name:
                prune_names.append(name)
                break

    # get Pruning class
    pruner = Pruning(
        net, device, prune_names, final_rate=args.final_rate, pruning_iter=args.pruning_iter
    )
    if args.init_masks is not None:
        print("taking init_masks")
        pruner.load(args.init_masks)
        pruner.apply_mask(pruner.remain_mask, pruner._model)
    
    model_params = [{'params': net.parameters(), 'lr': 0.0001}]
    # train_data_loader, test_data_loader = get_loader(args, config)
    cur_lr = config.learning_rate
    optimizer = optim.Adam(model_params, lr=cur_lr)
    global_step = 0
    steps_epoch = global_step // train_data_loader.__len__()
    if True:
        for prune_step in range(pruner.pruning_iter + 1):
            optimizer = optim.Adam(model_params, lr=cur_lr)
            factor = pruner.cur_rate / 100.0
            factor = 1.0
            res = 0
            for pg in optimizer.param_groups:
                pg["lr"] = factor * pg["lr"]
            # optimizer = optim.Adam(model_params, lr=cur_lr)
            res = 0
            # 定义早停法参数
            best_val_PSNR = float('-inf')  # 初始化验证集最佳损失为正无穷
            patience = 30  # 没有改善的验证轮次数
            counter = 0  # 记录连续没有改善的验证轮次数
            best_epoch = 0
            for epoch in range(steps_epoch, config.tot_epoch):
                counter = counter+1
                train_one_epoch(args)
                if (epoch + 1) % config.save_model_freq == 0:
                    if not os.path.exists(config.models + '/prune_step{}/'.format(prune_step)):
                        os.makedirs(config.models + '/prune_step{}/'.format(prune_step,task_name,config.filename, epoch + 1))
                    save_model(net, save_path=config.models + '/prune_step{}/{}_{}_EP{}.model'.format(prune_step,task_name,config.filename, epoch + 1))
                    
                res = test()
                # res = test_mul()
                logger.info("testing, Result: {} count: {}".format( res,counter))

                if res > best_val_PSNR :
                        best_val_PSNR = res
                        counter = 0        
                        best_epoch = epoch

                #         torch.save(
                #             net.state_dict(),
                #             os.path.join(
                #                 args.save_path,
                #                 "{}_epoch{}.th".format(task_name,epoch),
                #             ),
                #         )
                if counter==patience:
                    break
            with torch.no_grad():
                # 这部分是为了改变模型的形状
                for batch_idx, (input,snr) in enumerate(train_data_loader):
                    input = input.cuda()
                    snr_batch = torch.tensor(snr).expand(len(input)).unsqueeze(1).cuda()
                    net(input,snr_batch)
                    break
                    
            pruner.pruning_model()
                    
            pruner.save(
                os.path.join(
                    config.masks, "{}_final{}_epoch{}_{}_{}_noavgre{}.th".format(type,args.final_rate,best_epoch,pruner.prune_times, pruner.cur_rate,res)
                )
            )
            logger.info("No #{} testing, Result: {}".format(pruner.prune_times, res))
            
    else:
        PATH ="/home/zhoujh/WITT-main/saved_models/div_10-20dB_C96_final_ratio0.8/<class 'type'>_bestfinal0.8_epoch500_0_100.0_noavgre0.th"
        
        dumps = torch.load(PATH, map_location="cpu")
        # model =AE(c,h,w,P,device)
        net.load_state_dict(dumps, strict=False)
        test_only()