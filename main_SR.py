import torchvision.transforms
from PIL import Image
import torch
import my_utils
import SURE
import numpy as np

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GT_img = './baby.png'

scenarios = [1, 2, 3, 4, 5, 6]
print('Processing ' + GT_img[:-4])

I = Image.open(GT_img)

for scenario in scenarios:
    h, sig, P_eps, __, scale_factor = my_utils.get_SR_model(scenario)
    h = h / h.sum()
    h = h.to(device)

    T = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop((I.size[0] - I.size[0]%scale_factor, I.size[1] - I.size[1]%scale_factor)),
        torchvision.transforms.ToTensor()
    ])
    x = T(I).unsqueeze(0).to(device)

    with torch.no_grad():
        H = my_utils.fft_torch(h, (x.shape[2], x.shape[3]))
        y = my_utils.fft_Down_(x, H, scale_factor).real
        w = torch.randn_like(y) * sig
        y += w

    input_depth = x.shape[1]
    suffix = f'_SR_scenario_{scenario}_pGSURE'

    pad = 'circular'

    ###################################################################################################################
    # Define the network here
    from models import *
    NET_TYPE = 'skip' # UNet, ResNet
    net = get_net(input_depth, NET_TYPE, pad, n_channels=x.shape[1],
                  skip_n33d=128,
                  skip_n33u=128,
                  skip_n11=4,
                  num_scales=5,
                  upsample_mode='bilinear').to(device)
    net = torch.nn.DataParallel(net)
    ###################################################################################################################

    pGSURE = SURE.pGSURE_SR(h, y, scale_factor=scale_factor, sig2=sig ** 2, P_eps=P_eps)

    x_hat_best = None
    min_loss = np.inf

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    for i in range(10000):
        optimizer.zero_grad()

        loss = pGSURE.l_pGSURE_BP(net)

        loss.backward()
        optimizer.step()

        # Log
        with torch.no_grad():
            x_hat = net(pGSURE.u[0:1, :])
            MSE = torch.mean((x - x_hat) ** 2)
            PSNR_HR = -10*torch.log10(MSE)
            if loss < min_loss:
                x_hat_best = x_hat.clone()
                min_loss = loss.item()
        print(f'\r[GSURE - scenario {scenario}] Iteration {i} loss = {loss} PSNR_HR {PSNR_HR}', end='')
        if i % 100 == 0:
            torchvision.utils.save_image(x_hat_best, GT_img[:-4] + suffix + '.png')

    MSE = torch.mean((x - x_hat_best) ** 2)
    PSNR_HR = -20*torch.log10(MSE)
    print('\nBest PSNR, %.3f\n' % (PSNR_HR))
    torchvision.utils.save_image(x_hat_best, GT_img[:-4] + suffix + '.png')
