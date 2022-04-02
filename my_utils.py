import torch
import torch.fft
import numpy as np


def flip_torch(x):
    x_ = torch.flip(torch.roll(x, ((x.shape[2] // 2), (x.shape[3] // 2)), dims=(2, 3)), dims=(2, 3))
    return torch.roll(x_, (- (x_.shape[2] // 2), -(x_.shape[3] // 2)), dims=(2, 3))


def flip_np(x):
    x_ = np.flip(np.roll(x, ((x.shape[0] // 2), (x.shape[1] // 2)), (0, 1)))
    return np.roll(x_, (- (x_.shape[0] // 2), -(x_.shape[1] // 2)), (0, 1))


def fft_torch(x, s=None, zero_centered=True):
    # s = (Ny, Nx)
    __, __, H, W = x.shape
    if s == None:
        s = (H, W)
    if zero_centered:
        x_ = torch.roll(x, ((H // 2), (W // 2)), dims=(2, 3))
    else:
        x_ = x
    x_pad = torch.nn.functional.pad(x_, (0, s[1] - W, 0, s[0] - H))
    x_pad_ = torch.roll(x_pad, (- (H // 2), -(W // 2)), dims=(2, 3))
    return torch.fft.fftn(x_pad_, dim=(-2, -1))


def rfft_torch(x, s=None, zero_centered=True):
    # s = (Ny, Nx)
    __, __, H, W = x.shape
    if s == None:
        s = (H, W)
    if zero_centered:
        x_ = torch.roll(x, ((H // 2), (W // 2)), dims=(2, 3))
    else:
        x_ = x
    x_pad = torch.nn.functional.pad(x_, (0, s[1] - W, 0, s[0] - H))
    x_pad_ = torch.roll(x_pad, (- (H // 2), -(W // 2)), dims=(2, 3))
    return torch.fft.rfftn(x_pad_)


def bicubic_ker(x, y, a=-0.5):
    # X:
    abs_phase = np.abs(x)
    abs_phase2 = abs_phase ** 2
    abs_phase3 = abs_phase ** 3
    out_x = np.zeros_like(x)
    out_x[abs_phase <= 1] = (a + 2) * abs_phase3[abs_phase <= 1] - (a + 3) * abs_phase2[abs_phase <= 1] + 1
    out_x[(abs_phase > 1) & (abs_phase < 2)] = a * abs_phase3[(abs_phase > 1) & (abs_phase < 2)] - \
                                               5 * a * abs_phase2[(abs_phase > 1) & (abs_phase < 2)] + \
                                               8 * a * abs_phase[(abs_phase > 1) & (abs_phase < 2)] - 4 * a
    # Y:
    abs_phase = np.abs(y)
    abs_phase2 = abs_phase ** 2
    abs_phase3 = abs_phase ** 3
    out_y = np.zeros_like(y)
    out_y[abs_phase <= 1] = (a + 2) * abs_phase3[abs_phase <= 1] - (a + 3) * abs_phase2[abs_phase <= 1] + 1
    out_y[(abs_phase > 1) & (abs_phase < 2)] = a * abs_phase3[(abs_phase > 1) & (abs_phase < 2)] - \
                                               5 * a * abs_phase2[(abs_phase > 1) & (abs_phase < 2)] + \
                                               8 * a * abs_phase[(abs_phase > 1) & (abs_phase < 2)] - 4 * a

    return out_x * out_y


def sinc_ker(x, y):
    # X:
    out_x = np.zeros_like(x)
    out_x[x == 0] = 1
    out_x[x != 0] = np.sin(np.pi * x[x != 0]) / (np.pi * x[x != 0])

    out_y = np.zeros_like(y)
    out_y[y == 0] = 1
    out_y[y != 0] = np.sin(np.pi * y[y != 0]) / (np.pi * y[y != 0])

    return out_x * out_y


def build_flt(f, size):
    is_even_x = not size[1] % 2
    is_even_y = not size[0] % 2

    grid_x = np.linspace(-(size[1] // 2 - is_even_x * 0.5), (size[1] // 2 - is_even_x * 0.5), size[1])
    grid_y = np.linspace(-(size[0] // 2 - is_even_y * 0.5), (size[0] // 2 - is_even_y * 0.5), size[0])

    x, y = np.meshgrid(grid_x, grid_y)

    h = f(x, y)
    h = np.roll(h, (- (h.shape[0] // 2), -(h.shape[1] // 2)), (0, 1))

    return torch.tensor(h).float().unsqueeze(0).unsqueeze(0)


def get_bicubic(scale, size=None):
    f = lambda x, y: bicubic_ker(x / scale, y / scale)
    if size:
        h = build_flt(f, (size[0], size[1]))
    else:
        h = build_flt(f, (4 * scale + 8 + scale % 2, 4 * scale + 8 + scale % 2))
    return h


def get_box(supp, size=None):
    if size == None:
        size = (supp[0] * 2, supp[1] * 2)

    h = np.zeros(size)

    h[0:supp[0] // 2, 0:supp[1] // 2] = 1
    h[0:supp[0] // 2, -(supp[1] // 2):] = 1
    h[-(supp[0] // 2):, 0:supp[1] // 2] = 1
    h[-(supp[0] // 2):, -(supp[1] // 2):] = 1

    return torch.tensor(h).float().unsqueeze(0).unsqueeze(0)


def get_delta(size):
    h = torch.zeros(1, 1, size, size)
    h[0, 0, 0, 0] = 1
    return h


def get_gauss_flt(flt_size, std):
    f = lambda x, y: np.exp(-(x ** 2 + y ** 2) / 2 / std ** 2)
    h = build_flt(f, (flt_size, flt_size))
    return h


def get_sinc_flt(flt_size, W):
    f = lambda x, y: sinc_ker(W * x, W * y)
    h = build_flt(f, (flt_size, flt_size))
    return h


def fft_Filter_(x, A):
    X_fft = torch.fft.fftn(x, dim=(-2, -1))
    HX = A * X_fft
    return torch.fft.ifftn(HX, dim=(-2, -1))


def fft_Down_(x, H, alpha):
    X_fft = torch.fft.fftn(x, dim=(-2, -1))
    # H = fft_torch(h, s=X_fft.shape[2:4])
    HX = H * X_fft
    margin = (alpha - 1) // 2
    y = torch.fft.ifftn(HX, dim=(-2, -1))[:, :, margin:HX.shape[2] - margin:alpha, margin:HX.shape[3] - margin:alpha]
    return y


def fft_Down_freq_(X, H, alpha):
    HX = H * X
    margin = 0
    y = torch.fft.ifftn(HX, dim=(-2, -1))[:, :, margin:HX.shape[2] - margin:alpha, margin:HX.shape[3] - margin:alpha]
    return y


def fft_Up_(y, H, alpha):
    x = torch.zeros(y.shape[0], y.shape[1], y.shape[2] * alpha, y.shape[3] * alpha).to(y.device)
    # H = fft_torch(h, s=x.shape[2:4])
    start = alpha // 2
    x[:, :, start::alpha, start::alpha] = y
    X = torch.fft.fftn(x, dim=(-2, -1))
    HX = H * X
    return torch.fft.ifftn(HX, dim=(-2, -1))


def zero_SV(H, eps):
    abs_H = torch.abs(H)
    H[abs_H / abs_H.max() <= eps] = 0
    return H


def get_blur_model(scenario):
    if scenario == 1:
        f = lambda x, y: 1 / (1 + x ** 2 + y ** 2)
        h = build_flt(f, (15, 15))
        sigma = np.sqrt(2)
        P_eps = 5e-2
        ML_eps = P_eps
    elif scenario == 2:
        f = lambda x, y: 1 / (1 + x ** 2 + y ** 2)
        h = build_flt(f, (15, 15))
        sigma = np.sqrt(8)
        P_eps = 1e-1  # 5e-2
        ML_eps = P_eps
    elif scenario == 3:
        h = np.pad(np.ones((9, 9)), ((3, 3), (3, 3)))
        h = np.roll(h, (- (h.shape[0] // 2), -(h.shape[1] // 2)), (0, 1))
        h = torch.tensor(h).float().unsqueeze(0).unsqueeze(0)
        sigma = np.sqrt(0.3)
        P_eps = 5e-3
        ML_eps = P_eps
    elif scenario == 4:
        h_ = np.array([[1, 4, 6, 4, 1]])
        h_ = h_.T @ h_
        h = np.pad(h_, ((5, 5), (5, 5)))
        h = np.roll(h, (- (h.shape[0] // 2), -(h.shape[1] // 2)), (0, 1))
        h = torch.tensor(h).float().unsqueeze(0).unsqueeze(0)
        sigma = 7
        P_eps = 1e-1
        ML_eps = P_eps
    elif scenario == 5:
        h = get_gauss_flt((15, 15), 1.6)
        sigma = 2
        P_eps = 5e-2
        ML_eps = P_eps
    elif scenario == 6:
        h = get_gauss_flt((15, 15), 0.4)
        sigma = 8
        P_eps = 0
        ML_eps = P_eps
    elif scenario == 7:
        h = np.zeros((15, 15))
        h[3:-3, 3:-3] = np.ones((9, 9))
        h = np.roll(h, (- (h.shape[0] // 2), -(h.shape[1] // 2)), (0, 1))
        h = torch.tensor(h).float().unsqueeze(0).unsqueeze(0)
        sigma = 10
        P_eps = 1e-2
        ML_eps = P_eps

    return h, sigma / 255.0, P_eps, ML_eps


def get_SR_model(scenario):
    if scenario == 1:
        scale = 3
        h_std = 1.6
        f = lambda x, y: np.exp(-(x ** 2 + y ** 2) / 2 / h_std ** 2)
        h = build_flt(f, (15, 15))
        sigma = np.sqrt(10)
        P_eps = 1e-2
        ML_eps = P_eps
    elif scenario == 2:
        scale = 3
        h_std = 1.6
        f = lambda x, y: np.exp(-(x ** 2 + y ** 2) / 2 / h_std ** 2)
        h = build_flt(f, (15, 15))
        sigma = np.sqrt(49)
        P_eps = 1e-2
        ML_eps = P_eps
    elif scenario == 3:
        scale = 2
        f = lambda x, y: bicubic_ker(x / scale, y / scale)
        h = build_flt(f, (16, 16))
        sigma = np.sqrt(10)
        P_eps = 1e-2
        ML_eps = P_eps
    elif scenario == 4:
        scale = 3
        f = lambda x, y: bicubic_ker(x / scale, y / scale)
        h = build_flt(f, (15, 15))
        sigma = np.sqrt(10)
        P_eps = 0
        ML_eps = P_eps
    elif scenario == 5:
        scale = 2
        f = lambda x, y: bicubic_ker(x / scale, y / scale)
        h = build_flt(f, (16, 16))
        sigma = np.sqrt(49)
        P_eps = 0
        ML_eps = P_eps
    elif scenario == 6:
        scale = 3
        f = lambda x, y: bicubic_ker(x / scale, y / scale)
        h = build_flt(f, (15, 15))
        sigma = np.sqrt(49)
        P_eps = 0
        ML_eps = P_eps
    else:
        assert False

    return h, sigma / 255.0, P_eps, ML_eps, scale
