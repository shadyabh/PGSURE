from __future__ import print_function
import os

import torch
import torch.optim
import torchvision

import utils
import SURE

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =False


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', default='./data/Set5/', type=str)
parser.add_argument('--out_dir', default='./data/SR/', type=str)
parser.add_argument('--opt_suffix', default='', type=str)
parser.add_argument('--net_type', default='skip', type=str)
parser.add_argument('--lr', type=float, default=1e-2)
args = parser.parse_args()

in_dir = args.in_dir
out_dir = args.out_dir

imgs = [f for f in os.listdir(in_dir) if ('.png' in f)]
imgs.sort()

scenarios = [0,1,2,3,4,5]
# scenarios = [8,9,12,13]
PSNR_total = np.zeros((scenarios[-1] + 1))
PSNR_log = open(out_dir + 'log_GSURE_total_PSNR' + '.txt', 'w')

for img in imgs:
    print('Processing ' + img[:-4])
    path_to_image = in_dir + img

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_GT = True

    # Starts here
    GT = my_utils.load_img_torch(path_to_image, device)

    for scenario in scenarios:

        suffix = '_scenario_%d_' %scenario + args.net_type
        
        h, sig, P_eps, ML_eps, scale_factor = my_utils.get_SR_model(scenario)
        with torch.no_grad():
            x = GT.detach()
            if GT.shape[2]%scale_factor > 0:
                x = GT[:,:,:-(GT.shape[2]%scale_factor),:]
            if GT.shape[3]%scale_factor > 0:
                x = x[:,:,:, :-(GT.shape[3]%scale_factor)]
            h = h/h.sum()
            h = h .to(device)
        
        with torch.no_grad():
            H_flt = SURE.rfft_torch(h, (x.shape[2], x.shape[3]))
            h_ = torch.irfft(H_flt, signal_ndim=2, normalized=False, onesided=False)
            X = torch.rfft(x, signal_ndim=2, normalized=False, onesided=False)
            Y = SURE.fft_Down_(X, H_flt, scale_factor)
            y = torch.irfft(Y, signal_ndim=2, normalized=False, onesided=False)
            w = torch.randn_like(y)*sig
            y += w
            Y = torch.rfft(y, signal_ndim=2, normalized=False, onesided=False)

        H = lambda I: torch.irfft(SURE.fft_Down_(torch.rfft(I, signal_ndim=2, normalized=False, onesided=False), H_flt, scale_factor), signal_ndim=2, normalized=False, onesided=False)

        input_depth = x.shape[1]

        pad   =     'circular'

        NET_TYPE = args.net_type#'skip' # UNet, ResNet
        # suffix += '_' + NET_TYPE
        net = get_net(input_depth, NET_TYPE, pad, n_channels=x.shape[1],
                    skip_n33d=128, 
                    skip_n33u=128, 
                    skip_n11=4, 
                    num_scales=5,
                    upsample_mode='bilinear').to(device)

        
        # Losses
        if use_GT:
            pGSURE = SURE.pGSURE_SR(h, y, scale_factor, sig**2, GT=x, P_eps=P_eps, ML_eps=ML_eps)
        else:
            pGSURE = SURE.pGSURE_SR(h, y, sig**2, P_eps=P_eps, ML_eps=ML_eps)

        x_hat_best = None
        min_loss = np.inf

        def closure_GSURE():
            global i, min_loss, x_hat_best

            # with torch.no_grad():
            loss = pGSURE.l_pGSURE(net)
            # loss = pGSURE.l_pGSURE_BP(net)

            loss.backward()

            # Log
            with torch.no_grad():
                x_hat = net(pGSURE.u[0:1,:])
                MSEp = torch.sum( (pGSURE.P(x - x_hat))**2 ) /pGSURE.N
                MSE = torch.sum( (x - x_hat)**2 )/pGSURE.N
                psnr_HR = compare_psnr(torch_to_np(x), torch_to_np(x_hat))
                if loss < min_loss:# and pGSURE.div >= 0:
                    x_hat_best = x_hat
                    min_loss = loss.item()
            if i % 50 == 0:
                my_utils.save_img_torch(x_hat, out_dir + 'in_loop_pGSURE.png')
            print ('[GSURE - scenario %d] Iteration %05d estpMSE = %.7f pMSE = %.7f MSE = %.7f PSNR_HR %.3f' % (scenario, i, loss.item(), MSEp, MSE, psnr_HR))
            log.write('[GSURE] Iteration, %05d, estpMSE , %.7f, pMSE, %.7f, MSE, %.7f, PSNR_HR, %.3f, eps, %.13f\n' % (i, loss, MSEp, MSE, psnr_HR, pGSURE.eps))
            
            i += 1

            return loss

        psnr_history = [] 

        i = 0
        PSNR_DIP = 0
        OPT_OVER =  'net'
        p = get_params(OPT_OVER, net)
        LR = args.lr
        optimizer = None

        net = nn.DataParallel(net)
        pGSURE.H = nn.DataParallel(pGSURE.H)
        pGSURE.Ht = nn.DataParallel(pGSURE.Ht)
        pGSURE.HHt_dag = nn.DataParallel(pGSURE.HHt_dag)
        log = open(out_dir + img[:-4] + '_log_GSURE' + suffix + '.txt', 'w')
        # num_iter = 3000
        num_iter = 10000
        OPT_OVER = 'net'
        p = get_params(OPT_OVER, net)
        OPTIMIZER = 'adam'
        optimizer = optimize(OPTIMIZER, p, closure_GSURE, LR, num_iter, WD=0, moment=0, optimizer=optimizer)
        psnr_HR = compare_psnr(torch_to_np(x), torch_to_np(x_hat_best))
        PSNR_total[scenario] += psnr_HR/len(imgs)
        print('Best PSNR, %.3f\n' % (psnr_HR))
        log.write('Best PSNR, %.3f\n' % (psnr_HR))
        log.close()

        with torch.no_grad():
            my_utils.save_img_torch(x_hat_best, out_dir + img[:-4] + suffix + '_pGSURE.png')

PSNR_log.write(str(PSNR_total))
PSNR_log.close()
