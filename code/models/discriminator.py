import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, (3,3), stride=stride)
        self.conv = torch.nn.utils.spectral_norm(self.conv)
        self.lrelu = nn.LeakyReLU()
    def forward(self, x):
        x = self.lrelu(self.conv(x))
        return x
   
    
class Discriminator(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.conv_blocks = nn.Sequential(ConvBlock(in_channels, 8, 2),
                                         ConvBlock(8, 16, 2),
                                         ConvBlock(16, 32, 2),
                                         ConvBlock(32, 64, 2),
                                         ConvBlock(64, 128, 1),
                                         ConvBlock(128, 256, 1),
                                         nn.Conv2d(256, 1, 1))
        self.act_last = nn.Sigmoid()
        
    def forward(self, x):
        """x: (B,C,T,F)"""
        fmap = []
        for i, conv_block in enumerate(self.conv_blocks):
            x = conv_block(x)
            if i < len(self.conv_blocks):
                fmap.append(x)
        x = self.act_last(x)
        return x, fmap
    
    
class MultiResolutionDiscriminator(nn.Module):
    def __init__(self,
                fft_sizes=[2048, 1024, 512],
                hop_sizes=[240, 120, 50],
                win_lengths=[1200, 600, 240]):
        super().__init__()
        self.mrd = nn.ModuleList()
        self.fft_sizes = fft_sizes 
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        for _ in fft_sizes:
            self.mrd += [Discriminator(in_channels=3)]
            
    def forward(self, x):
        """x: (B, T), time signal"""
        y_lst = []
        fmap_lst = []
        for fs, hs, wl, model in zip(self.fft_sizes, self.hop_sizes, self.win_lengths, self.mrd):
            x_spec = torch.stft(x, fs, hs, wl, torch.ones(wl).to(x.device), return_complex=True)  # (B,F,T)
            x_lps = torch.log10(torch.abs(x_spec).clamp(1e-8))[...,None]
            x_input = torch.cat([x_lps, torch.view_as_real(x_spec)], dim=-1).permute(0,3,2,1)  # (B,3,T,F)
            y, fmap = model(x_input)  # (x, fmap)
            y_lst.append(y)
            fmap_lst.append(fmap)
            
        return y_lst, fmap_lst
    

class MultiBandDiscriminator(nn.Module):
    def __init__(self, n_bands=3):
        super().__init__()
        self.bs = BandSplit(n_bands=n_bands)
        self.mbd = torch.nn.ModuleList()
        for _ in range(n_bands):
            self.mbd += [Discriminator(in_channels=2)]
    def forward(self, x):
        """
        x: (B,F,T,2)
        """
        y_lst = []
        fmap_lst = []
        x_bands = self.bs(x.permute(0,3,2,1))
        for i, model in enumerate(self.mbd):
            x_subband = x_bands[:, i*2:(i+1)*2, :, :]
            y, fmap = model(x_subband)
            y_lst.append(y)
            fmap_lst.append(fmap)
            
        return y_lst, fmap_lst
    
    
class BandSplit(nn.Module):
    def __init__(self, n_freqs=769, n_bands=3):
        super().__init__()
        self.n_freqs = n_freqs
        self.n_bands = n_bands
    
    def forward(self, x):
        """x: (B,C,T,F)"""
        assert x.shape[-1] == self.n_freqs
        sub_nfreqs = self.n_freqs // self.n_bands + 1
        x_bands = []
        for i in range(self.n_bands):
            x_bands.append(x[..., i*(sub_nfreqs-1): i*(sub_nfreqs-1)+sub_nfreqs])
        x_bands = torch.cat(x_bands, dim=1)
        return x_bands
    
    def inverse(self, x_bands):
        """x_bands: (B,C*n,T,F)"""
        assert x_bands.shape[-1] == self.n_freqs // self.n_bands + 1
        x = [x_bands[:, :2]]
        for i in range(1, self.n_bands):
            x.append(x_bands[:, i*2:(i+1)*2, :, 1:])
        return torch.cat(x, dim=-1)
    
        
class MultiDiscriminator(nn.Module):
    def __init__(self, 
                 n_bands = 3, 
                 fft_sizes=[2048, 1024, 512],
                hop_sizes=[240, 120, 50],
                win_lengths=[1200, 600, 240]):
        super(MultiDiscriminator, self).__init__()
        self.n_bands  = n_bands 
        self.mrd = MultiResolutionDiscriminator(fft_sizes, hop_sizes, win_lengths)
        self.mbd = MultiBandDiscriminator(n_bands)
    def forward(self, x_time, x_spec):
        y_lst_mrd, fmap_lst_mrd = self.mrd(x_time)
        y_lst_mbd, fmap_lst_mbd = self.mbd(x_spec)
        return y_lst_mrd + y_lst_mbd, fmap_lst_mrd + fmap_lst_mbd
    

if __name__ == "__main__":
    model = MultiDiscriminator()
    x_time = torch.randn(1, 48000*2)
    x_spec = torch.randn(1, 769, 252, 2)
    y_lst, fmap_lst = model(x_time, x_spec)
    
    for y in y_lst:
        print(y.shape)
    
    for fmap in fmap_lst:
        for item in fmap:
            print(item.shape, end="\t")
        print("\n")
        
            

        