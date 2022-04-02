import torch
import numpy as np
import my_utils

def dagger(H, method='Naive', eps=1e-3):
    abs_H = torch.abs(H)
    if method == 'Naive':
        H_pinv = torch.zeros_like(H)
        H_pinv[(abs_H / abs_H.max()) > 0] = 1 / H[(abs_H / abs_H.max()) > 0]
    else:
        H_pinv = H.conjugate() / (abs_H ** 2 + eps ** 2)
    return H_pinv


def zero_SV(H, eps):
    abs_H = np.abs(H)
    H[abs_H / abs_H.max() <= eps] = 0
    return H

def zero_SV_torch(H, eps):
    abs_H = torch.abs(H)
    H[abs_H / abs_H.max() <= eps] = 0
    return H


def shift_by(H, shift):
    k_x = torch.linspace(0, H.shape[1] - 1, H.shape[1])
    k_y = torch.linspace(0, H.shape[0] - 1, H.shape[0])

    exp_x, exp_y = torch.meshgrid(np.exp(-1j * 2 * np.pi * k_x * shift / H.shape[1]),
                               np.exp(-1j * 2 * np.pi * k_y * shift / H.shape[0]))

    return H * exp_x * exp_y


def fft_np(x, s):
    # s = (Ny, Nx)
    H,W = x.shape
    x_ = np.roll(x, ((H//2), (W//2)), axis=(0,1))
    x_pad = np.pad(x_, ((0, s[0] - H), (0, s[1] - W)))
    x_pad_ = np.roll(x_pad, (- (H//2), -(W//2)), axis=(0,1))
    return np.fft.fft2(x_pad_)

class pGSURE_blur():
    def __init__(self, h, y, sig2, GT=None, P_eps=0):
        super(pGSURE_blur, self).__init__()
        self.sig2 = sig2
        self.y = y
        self.N = y.view(-1).shape[0]
        with torch.no_grad():
            h_np = np.array(h.clone().cpu())[0, 0, :, :]
            self.h = h_np
            H = fft_np(h_np, s=(y.shape[2], y.shape[3]))
            H = zero_SV(H, P_eps)
            H_ = torch.tensor(H).float().unsqueeze(0).unsqueeze(0).cuda()
            self.H = lambda I: my_utils.fft_Filter_(I, H_).real
            self.Ht = self.H
            Ht = fft_np(my_utils.flip_np(h_np), s=(y.shape[2], y.shape[3]))
            Ht = zero_SV(Ht, P_eps)

            HtH_dag = dagger(Ht * H)
            Ht_HtH_dag_H_np = Ht * HtH_dag * H
            Ht_HtH_dag_H = torch.tensor(np.imag(Ht_HtH_dag_H_np)).float().cuda()

            self.P = lambda I: my_utils.fft_Filter_(I, Ht_HtH_dag_H).real

            u = self.Ht(y)

            HtH_dag = dagger(Ht * H)
            HtH_dag_Ht_np = HtH_dag * Ht
            HtH_dag_Ht = torch.tensor(HtH_dag_Ht_np).float().cuda()
            self.HtH_dag_Ht = lambda I: my_utils.fft_Filter_(I, HtH_dag_Ht).real
            self.x_ML = my_utils.fft_Filter_(y, HtH_dag_Ht).real
            self.fu = None
            self.div = None
            self.u = u.repeat(torch.cuda.device_count(), 1, 1, 1)
            self.eps = 1e-6
            self.GT = GT
            self.c1, self.c2, self.c3, self.c4 = None, None, None, None

    def div_approx(self, f):
        b = torch.randn_like(self.u)
        m = b.mean()
        c = torch.sqrt(torch.mean((b - m) ** 2))
        b = (b - m) / c
        df = self.P(f(self.u + self.eps * b) - self.fu)
        N_eps = self.eps * self.N
        MC = torch.mean(torch.sum(b * df / N_eps, dim=(1, 2, 3)), dim=0)

        self.div = MC
        return MC

    def l_pGSURE(self, f):
        self.fu = f(self.u)
        if self.GT != None:
            self.c1 = torch.mean(self.P(self.GT) ** 2)
        else:
            self.c1 = 1
        self.c2 = torch.mean((self.P(self.fu)) ** 2)
        self.c3 = self.sig2 * self.div_approx(f)
        self.c4 = torch.mean(self.x_ML * self.fu)

        eta = self.c1 + self.c2 - 2 * self.c4 + 2 * self.c3

        return eta

    def bp(self):
        LS = self.H(self.fu) - self.y
        l = torch.mean(self.HtH_dag_Ht(LS) ** 2)
        return l

    def l_pGSURE_BP(self, f):
        self.fu = f(self.u)
        self.c3 = self.sig2 * self.div_approx(f)

        eta = self.bp() + 2 * self.c3

        return eta


class pGSURE_denoise():
    def __init__(self, y, sig2=1, GT=None):
        super(pGSURE_denoise, self).__init__()
        self.sig2 = sig2
        self.N = y.view(-1).shape[0]
        self.y = y
        u = y

        self.x_ML = y
        self.fu = None
        self.div = None
        self.u = u.repeat(5 * torch.cuda.device_count(), 1, 1, 1)
        self.eps = 1e-3
        self.GT = GT

    def div_approx(self, f):
        b = torch.randn_like(self.u)
        m = b.mean()
        c = torch.sqrt(torch.mean((b - m) ** 2))
        b = (b - m) / c
        df = f(self.u + self.eps * b) - (self.fu).repeat(self.u.shape[0], 1, 1, 1)
        N_eps = self.eps * self.N
        MC = torch.mean(torch.sum(b * df / N_eps, dim=(1, 2, 3)), dim=0)

        self.div = MC
        return MC

    def l_pGSURE(self, f):
        self.fu = f(self.u[0:1, :])
        return torch.mean((self.y - self.fu) ** 2) + 2 * self.sig2 * self.div_approx(f)


class pGSURE_SR():
    def __init__(self, h, y, scale_factor, sig2=1, P_eps=0):
        super(pGSURE_SR, self).__init__()
        self.sig2 = sig2
        self.N = y.reshape(-1).shape[0] * scale_factor * scale_factor
        self.y = y
        self.scale_factor = scale_factor
        with torch.no_grad():
            h_np = np.array(h.clone().cpu())[0, 0, :, :]
            B = fft_np(h_np / h_np.sum(), s=(y.shape[2] * scale_factor, y.shape[3] * scale_factor))

            Bt = fft_np(my_utils.flip_np(h_np / h_np.sum()), s=(y.shape[2] * scale_factor, y.shape[3] * scale_factor))

            B = zero_SV(B, P_eps)
            Bt = zero_SV(Bt, P_eps)

            B = torch.tensor(B).float().unsqueeze(0).unsqueeze(0).cuda()
            Bt = torch.tensor(Bt).float().unsqueeze(0).unsqueeze(0).cuda()
            self.H = lambda I: my_utils.fft_Down_(I, B, scale_factor).real
            self.Ht = lambda I: my_utils.fft_Up_(I, Bt, scale_factor).real

            if scale_factor % 2:
                HHt = self.H(Bt)
            else:
                HHt = my_utils.fft_Up_(shift_by(Bt, 0.5), shift_by(B, 0.5), scale_factor)

            HHt_ = zero_SV_torch(HHt, P_eps)
            HHt_dag = dagger(HHt_)
            self.HHt_dag = lambda I: my_utils.fft_Filter_(I, HHt_dag).real

            self.P = lambda I: self.Ht(self.HHt_dag(self.H(I)))

            self.x_ML = self.Ht(self.HHt_dag(y))

            u = self.Ht(y)

            self.fu = None
            self.div = None
            self.u = u.repeat(torch.cuda.device_count(), 1, 1, 1)
            self.eps = 1e-6
            self.c1, self.c2, self.c3, self.c4 = None, None, None, None
            self.scale_factor = scale_factor

    def div_approx(self, f):
        b = torch.randn_like(self.u)
        m = b.mean()
        c = torch.sqrt(torch.mean((b - m) ** 2))
        b = (b - m) / c
        df = self.P(f(self.u + self.eps * b) - self.fu)
        N_eps = self.eps * self.N
        MC = torch.mean(torch.sum(b * df / N_eps, dim=(1, 2, 3)), dim=0)

        self.div = MC
        return MC

    def bp(self, f):
        fu = f(self.u[0:1, :])
        LS = self.H(fu) - self.y
        l = torch.mean(self.Ht(self.HHt_dag(LS)) ** 2)
        return l

    def l_pGSURE_BP(self, f):
        self.fu = f(self.u)
        self.c3 = self.sig2 * self.div_approx(f)

        eta = self.bp(f) + 2 * self.c3

        return eta