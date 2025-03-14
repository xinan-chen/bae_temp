import toml
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as Func
from utils import PQMF

# config = toml.load('cfg_urgent.toml')


# class loss_mse(nn.Module):
#     def __init__(self):
#         super(loss_mse, self).__init__()
#         self.register_buffer("windows", torch.hann_window(config['FFT']['win_length']))
#         self.nfreq = self.win_len//2 + 1
#         self.mse_loss = nn.MSELoss(reduction='sum')

#     def forward(self, est, clean):
#         data_len = min(est.shape[-1], clean.shape[-1])
#         est = est[..., :data_len]
#         clean = clean[..., :data_len]

#         est_stft = torch.stft(est, **config['FFT'], center=True, window=self.windows, return_complex=False)   
#         clean_stft = torch.stft(clean, **config['FFT'], center=True, window=self.windows, return_complex=False)
#         est_stft_real, est_stft_imag = est_stft[:,:,:,0], est_stft[:,:,:,1]
#         clean_stft_real, clean_stft_imag = clean_stft[:,:,:,0], clean_stft[:,:,:,1]
#         est_mag = torch.sqrt(est_stft_real**2 + est_stft_imag**2 + 1e-12)
#         clean_mag = torch.sqrt(clean_stft_real**2 + clean_stft_imag**2 + 1e-12)
#         est_real_c = est_stft_real / (est_mag**(0.7))
#         est_imag_c = est_stft_imag / (est_mag**(0.7))
#         clean_real_c = clean_stft_real / (clean_mag**(0.7))
#         clean_imag_c = clean_stft_imag / (clean_mag**(0.7))

#         mag_loss = 0.7 * self.mse_loss(est_mag**(0.3), clean_mag**(0.3)) + \
#                 0.3 * (self.mse_loss(est_real_c, clean_real_c) + \
#                 self.mse_loss(est_imag_c, clean_imag_c))
        
#         return mag_loss / est.shape[0]


class loss_hybrid(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_stft, true_stft):
        """
        pred_stft: (B,F,T,2), B is batch size, F is frequency, T is time, the last dimension is real/imaginary component
        true_stft: (B,F,T,2)
        """
        device = pred_stft.device

        pred_stft_real, pred_stft_imag = pred_stft[:,:,:,0], pred_stft[:,:,:,1]
        true_stft_real, true_stft_imag = true_stft[:,:,:,0], true_stft[:,:,:,1]
        pred_mag = torch.sqrt(pred_stft_real**2 + pred_stft_imag**2 + 1e-12)
        true_mag = torch.sqrt(true_stft_real**2 + true_stft_imag**2 + 1e-12)
        pred_real_c = pred_stft_real / (pred_mag**(0.7))
        pred_imag_c = pred_stft_imag / (pred_mag**(0.7))
        true_real_c = true_stft_real / (true_mag**(0.7))
        true_imag_c = true_stft_imag / (true_mag**(0.7))
        real_loss = nn.MSELoss()(pred_real_c, true_real_c)
        imag_loss = nn.MSELoss()(pred_imag_c, true_imag_c)
        mag_loss = nn.MSELoss()(pred_mag**(0.3), true_mag**(0.3))
        
        # sisnr
        nfft = (true_stft.shape[1]-1) * 2
        hop_len = nfft // 2
        y_pred = torch.istft(pred_stft_real+1j*pred_stft_imag, nfft, hop_len, nfft, window=torch.hann_window(nfft).to(device))
        y_true = torch.istft(true_stft_real+1j*true_stft_imag, nfft, hop_len, nfft, window=torch.hann_window(nfft).to(device))
        y_true = torch.sum(y_true * y_pred, dim=-1, keepdim=True) * y_true / (torch.sum(torch.square(y_true),dim=-1,keepdim=True) + 1e-8)

        sisnr =  - torch.log10(torch.norm(y_true, dim=-1, keepdim=True)**2 / (torch.norm(y_pred - y_true, dim=-1, keepdim=True)**2 + 1e-8) + 1e-8).mean()
        
        # return real_loss, imag_loss, mag_loss, sisnr
        # print(real_loss, imag_loss, mag_loss, sisnr)
        return 30*(real_loss + imag_loss) + 70*mag_loss + sisnr


class loss_hybrid_pl(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = loss_hybrid()

    def forward(self, pred_stft, true_stft):
        """
        pred_stft: (B,6,F,T,2)
        true_stft: (B,6,F,T,2)
        """
        loss = 0
        for ii in range(pred_stft.shape[1]):
            loss += self.loss(pred_stft[:,ii], true_stft[:,ii])
        loss = loss / (ii+1)
        return loss


class STFTLoss(nn.Module):
    def __init__(self, n_fft=1024, hop_len=120, win_len=600, window="hann_window"):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.register_buffer("window", getattr(torch, window)(win_len))

    def loss_spectral_convergence(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")

    def loss_log_magnitude(self, x_mag, y_mag):
        return Func.l1_loss(torch.log(y_mag), torch.log(x_mag))

    def forward(self, x, y):
        """x, y: (B, T), in time domain"""
        x = torch.stft(x, self.n_fft, self.hop_len, self.win_len, self.window.to(x.device), return_complex=True)
        y = torch.stft(y, self.n_fft, self.hop_len, self.win_len, self.window.to(x.device), return_complex=True)
        x_mag = torch.abs(x).clamp(1e-8)
        y_mag = torch.abs(y).clamp(1e-8)
        
        sc_loss = self.loss_spectral_convergence(x_mag, y_mag)
        mag_loss = self.loss_log_magnitude(x_mag, y_mag)
        # ri_loss = self.loss_real_imag(torch.view_as_real(x), torch.view_as_real(y))
        # print(sc_loss, mag_loss, 200*ri_loss)
        loss = sc_loss + mag_loss
        return loss


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes=[2048, 1024, 512],
        hop_sizes=[240, 120, 50],
        win_lengths=[1200, 600, 240],
        window="hann_window",
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = nn.ModuleList()
        for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, hs, wl, window)]

    def forward(self, x, y):
        loss = 0.0
        for f in self.stft_losses:
            loss += f(x, y)
        loss /= len(self.stft_losses)
        return loss


class GesperLoss(nn.Module):
    def __init__(
        self,
        fft_sizes=[2048, 1024, 512],
        hop_sizes=[240, 120, 50],
        win_lengths=[1200, 600, 240],
        window="hann_window",
        n_bands=3
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.n_bands = n_bands
        self.pqmf = PQMF(n_bands)
        sub_fft_sizes = [item // n_bands for item in fft_sizes]
        sub_hop_sizes = [item // n_bands for item in hop_sizes]
        sub_win_lengths = [item // n_bands for item in win_lengths]
        self.multiR_stft_loss = MultiResolutionSTFTLoss(fft_sizes, hop_sizes, win_lengths, window)
        self.sub_multiR_stft_loss = MultiResolutionSTFTLoss(sub_fft_sizes, sub_hop_sizes, sub_win_lengths, window)

    def forward(self, x, y):
        loss = 0.0
        xs = self.pqmf.analysis(x[:,None])
        ys = self.pqmf.analysis(y[:,None])
        for i in range(self.n_bands):
            loss += self.sub_multiR_stft_loss(xs[:,i], ys[:,i])
        loss /= self.n_bands
        loss += self.multiR_stft_loss(x, y)
        return loss
    

# def LSDLoss(esti, clean):  # 2~4
#     """spec input"""
#     esti_mag = torch.norm(esti, dim=-1)
#     clean_mag = torch.norm(clean, dim=-1)
#     esti_log = torch.log10(torch.clamp(esti_mag, min=1e-8))
#     clean_log = torch.log10(torch.clamp(clean_mag, min=1e-8))
#     lsd = torch.mean((esti_log - clean_log) ** 2, dim=-1)
#     lsd = torch.mean(torch.sqrt(lsd))
#     return lsd

def LSDLoss(esti, clean, Fdim=1):  # 0,1,2,3
    esti = torch.stft(esti, 2048, 512, 2048, torch.hann_window(2048).to(esti.device), return_complex=True)
    clean = torch.stft(clean, 2048, 512, 2048, torch.hann_window(2048).to(esti.device), return_complex=True)
    """spec : (B,C,F,T,2)"""
    esti_mag = torch.abs(esti).square()
    clean_mag = torch.abs(clean).square()
    esti_log = torch.log10(torch.clamp(esti_mag, min=1e-8))
    clean_log = torch.log10(torch.clamp(clean_mag, min=1e-8))
    lsd = torch.sqrt(torch.mean((esti_log - clean_log) ** 2, dim=Fdim))
    lsd = torch.mean(lsd)
    return lsd


def MCDLoss(x, y, fs, device):  # 0.2~0.5
    """time domain input"""
    assert fs == 48000
    n_mfcc = get_best_mcep_params(fs)[0]
    kwargs = {
                "n_mels": 120,
                "n_fft": 2048,
                "win_length": 1200,
                "hop_length": 240,
                "window_fn": torch.hamming_window,
            }

    esti_mfcc = torchaudio.transforms.MFCC(sample_rate=fs, n_mfcc=n_mfcc, melkwargs=kwargs).to(device)(x)
    true_mfcc = torchaudio.transforms.MFCC(sample_rate=fs, n_mfcc=n_mfcc, melkwargs=kwargs).to(device)(y)
    mcd = torch.mean((esti_mfcc - true_mfcc)**2)
    return mcd


def get_best_mcep_params(fs):
    # https://sp-nitech.github.io/sptk/latest/main/mgcep.html#_CPPv4N4sptk19MelCepstralAnalysisE
    if fs == 8000:
        return 13, 0.31
    elif fs == 16000:
        return 23, 0.42
    elif fs == 22050:
        return 34, 0.45
    elif fs == 24000:
        return 34, 0.46
    elif fs == 32000:
        return 36, 0.50
    elif fs == 44100:
        return 39, 0.53
    elif fs == 48000:
        return 39, 0.55
    else:
        raise ValueError(f"Not found the setting for {fs}.")        


if __name__=='__main__':
    a = torch.randn(2, 48000*2)
    b = torch.randn(2, 48000*2)
    
    loss_gesper = GesperLoss(sr=48000)
    loss = loss_gesper(a, b)
    print(loss)
