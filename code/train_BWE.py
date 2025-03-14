# warning: the code can't run DDP
import warnings
warnings.filterwarnings("ignore")
import os
import sys
# sys.path.append('/home/nis/xinan.chen/projects/BWE/')
import toml
import torch
import shutil
import random
import argparse
import numpy as np
import torch.distributed as dist
from datetime import datetime
from tqdm import tqdm
from glob import glob
from pathlib import Path
from pesq import pesq
import soundfile as sf
from torch.utils.tensorboard import SummaryWriter
from distributed_utils import reduce_value

from models.generator_bae_lite import Generator as Generator # change!!!
from models.discriminator_msstft_ygc import Discriminator_stft as Discriminator
from dataloader_BWE import VctkDataset as VctkDataset # change!!!
from utils import LWCA2 as WarmupLR
from loss import GesperLoss, LSDLoss, MCDLoss
from stft_loss_48k import MultiResolutionSTFTLoss, LSD
from utils import hold_gpu_memory
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback,RichProgressBar,LearningRateMonitor

# torch.backends.cudnn.deterministic =True
class NaNMonitor(Callback):
    def __init__(self, checkpoint_dir):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir

    def on_train_epoch_end(self, trainer, pl_module):
        # 检查是否有 NaN 值
        if pl_module.checknan:
            print("NaN detected! Restoring from the latest checkpoint...")
            self.latest_checkpoint = sorted(glob(os.path.join(self.checkpoint_dir, 'model_*.tar')))[-1]
            if self.latest_checkpoint:
                pl_module.load_from_checkpoint(self.latest_checkpoint)
            else:
                print("No checkpoint found. Training will be stopped.")
                trainer.should_stop = True

def run(config, args):
    args.device = torch.device('cuda')
    train_dataset = VctkDataset(**config['train_dataset'], **config['FFT'])
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, **config['train_dataloader'],
                                                    shuffle=True)
    validation_dataset = VctkDataset(**config['validation_dataset'], **config['FFT'])
    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, **config['validation_dataloader'],
                                                            shuffle=False)
        
    model = Generator(**config['bwe_config']).to(args.device)
    model_d = Discriminator().to(args.device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4, betas=(0.8, 0.99))
    optimizer_d = torch.optim.Adam(params=model_d.parameters(), lr=5e-5, betas=(0.8, 0.99))

    trainer_config = config['trainer']
    resume = trainer_config['resume']
    if not resume:
        exp_path = trainer_config['exp_path'] + '_' + datetime.now().strftime("%Y-%m-%d-%Hh%Mm")

    else:
        exp_path = trainer_config['exp_path'] + '_' + trainer_config['resume_datetime']
        if not os.path.exists(exp_path):
            raise FileNotFoundError('The specified experiment path does not exist.')
    log_path = os.path.join(exp_path, 'logs')
    checkpoint_path = os.path.join(exp_path, 'checkpoints')
    # sample_path = os.path.join(exp_path, 'val_samples')
    code_path = os.path.join(exp_path, 'code')

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    # os.makedirs(sample_path, exist_ok=True)
    os.makedirs(code_path, exist_ok=True)

    # save the config
    with open(
        os.path.join(
            exp_path, 'config.toml'.format(datetime.now().strftime("%Y-%m-%d-%Hh%Mm"))), 'w') as f:
        toml.dump(config, f)
    shutil.copy2(__file__, exp_path)
    for file in Path(__file__).parent.iterdir():
        if file.is_file():
            shutil.copy2(file, code_path)
    shutil.copytree(Path(__file__).parent / 'models', Path(code_path) / 'models', dirs_exist_ok=True)



    BWEmodel = BWEModel(config=config, model=[model, model_d],optimizer=[optimizer, optimizer_d], loss_func=None,
                      train_dataloader=train_dataloader, validation_dataloader=validation_dataloader, 
                      train_sampler=None, args=args)
    
    trainer = pl.Trainer(
        max_epochs=config['trainer']['max_epochs'],
        logger=pl.loggers.TensorBoardLogger(log_path, name='version_0'),
        check_val_every_n_epoch=1,
        callbacks=[
            pl.callbacks.ModelCheckpoint(dirpath = checkpoint_path,
                                         filename='model_{epoch}',
                                         save_last=True, 
                                         monitor='val_loss_lsd', 
                                         save_top_k=1, 
                                         mode='min'),
            LearningRateMonitor(logging_interval='step'),
            RichProgressBar(leave=False),
            NaNMonitor(checkpoint_path)
        ],
    )
    if resume:
        # latest_checkpoint = sorted(glob(os.path.join(checkpoint_path, 'model_*.tar')))[-1]
        trainer.fit(
            BWEmodel,
            ckpt_path=os.path.join(checkpoint_path, 'last.ckpt'),
        )
    else:
        trainer.fit(BWEmodel)



class BWEModel(pl.LightningModule):
    def __init__(self, config, model, optimizer, loss_func,
                 train_dataloader, validation_dataloader, train_sampler, args):
        super(BWEModel, self).__init__()
        self.config = config
        self.automatic_optimization = False

        self.model = model[0]
        self.model_d = model[1]
        self.optimizer = optimizer[0]
        self.optimizer_d = optimizer[1]
        # self.scheduler = WarmupLR(self.optimizer, **config['scheduler'])
        # self.scheduler_d = WarmupLR(self.optimizer_d, **config['schedulerd'])
        # self.loss_func = loss_func

        self.training_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

        # self.train_sampler = train_sampler
        # self.device = args.device
        # training config
        # self.lamda = config['coeff']['lamda']
        # self.lamda_f = config['coeff']['lamda_f']
        # self.lamda_g = config['coeff']['lamda_g']
        self.clip_grad_norm_value = config['trainer']['clip_grad_norm_value']

        self.loss_func = MultiResolutionSTFTLoss()
        self.wav_loss = torch.nn.MSELoss()
        self.checknan = False
        self.fs =config['FFT']['fs']
        self.fft_len = config['FFT']['fft_len']
        self.hop_len = config['FFT']['hop_len']
        self.win_len = config['FFT']['win_len']
        self.low_fs = config['FFT']['low_fs']
        self.win = torch.hann_window(self.win_len).cuda()

    def configure_optimizers(self):

        optimizer_g = self.optimizer
        optimizer_d = self.optimizer_d
        return [optimizer_g, optimizer_d], []

    def loss_feat(self, esti_fmap, true_fmap):
        loss = 0
        for (de, dt) in zip(esti_fmap, true_fmap):
            for (e, t) in zip(de, dt):
                loss += torch.mean(torch.abs(e - t))/len(esti_fmap)/len(de)
        return loss
    
    def loss_adv(self, esti_metric, true_metric):
        loss = 0
        for de, dt in zip(esti_metric, true_metric):
            e_loss = torch.mean(torch.clamp(1-de[-1],min=0))/len(esti_metric)
            loss += e_loss
        return loss
    
    def loss_dis(self, esti_metric, true_metric):
        loss = 0
        for de, dt in zip(esti_metric, true_metric):
            e_loss = torch.mean(torch.clamp(1+de[-1],min=0))/len(esti_metric)
            t_loss = torch.mean(torch.clamp(1-dt[-1],min=0))/len(esti_metric)
            loss += (t_loss + e_loss)
        return loss
    
    def forward(self, clean, ifloss=True):
        clean = clean.to(self.device) # (B,T)
        # stft
        clean_spec = torch.stft(clean, self.fft_len, self.hop_len, self.win_len, self.win, return_complex=True) # (B,F,T)
        clean_spec_mag = torch.abs(clean_spec) ** 0.5 # (B,F,T)
        clean_spec_phase = torch.angle(clean_spec) # (B,F,T)
        freqbin = self.win_len // 2 + 1
        low_freqbin = (self.win_len // 2) * self.low_fs // self.fs +1
        input_spec_mag = clean_spec_mag.clone()
        input_spec_mag[:,low_freqbin:,:] = 0

        esti_spec_mag = self.model(input_spec_mag)

        flip_spec_phase1 = torch.flip(clean_spec_phase[:,:(low_freqbin-1),:], [1])
        flip_spec_phase2 = torch.cat((clean_spec_phase[:,:low_freqbin,:], -1* flip_spec_phase1,-1* flip_spec_phase1), dim=1)
        # flip_spec_phase3 = torch.flip(flip_spec_phase2[:,:2*(low_freqbin-1),:], [1])
        # flip_spec_phase4 = torch.cat((flip_spec_phase2, -1* flip_spec_phase3), dim=1)
        # flip_spec_phase = flip_spec_phase4[:,:freqbin,:]
        flip_spec_phase = flip_spec_phase2
        
        enhanced_real = (esti_spec_mag ** 2) * torch.cos(flip_spec_phase)
        enhanced_imag = (esti_spec_mag ** 2) * torch.sin(flip_spec_phase)
        enhanced = torch.istft(enhanced_real + 1j * enhanced_imag, self.fft_len, self.hop_len, self.win_len, self.win)
        ## For generator
        
        if ifloss:
            # wavform_loss
            loss_wav = 100 * self.wav_loss(enhanced, clean)
            # stft_loss
            loss_stft =10* self.loss_func(enhanced, clean)
            # adv_loss
            esti_fmap = self.model_d(enhanced)
            true_fmap = self.model_d(clean)
            loss_adv = self.loss_adv(esti_fmap, true_fmap)
            # feat_loss
            loss_feat =10 * self.loss_feat(esti_fmap, true_fmap)
        
            # loss_lsd = 5 * LSDLoss(enhanced, clean)
            # loss_mcd = 1/20 * MCDLoss(enhanced, clean, 48000, self.device)
            # loss_gesper = self.loss_func(enhanced, clean)

            loss = loss_wav + loss_stft + loss_adv + loss_feat
            if torch.isnan(loss):
                self.checknan = True
        else:
            loss = loss_wav = loss_stft = loss_adv = loss_feat = 0

        return loss, loss_wav, loss_stft, loss_adv, loss_feat, enhanced, clean, 


    def training_step(self, batch, batch_idx):
        optimizer_g, optimizer_d = self.optimizers()
        self.toggle_optimizer(optimizer_g)
        loss, loss_wav, loss_stft, loss_adv, loss_feat, enhanced, clean,= self.forward(batch)
        self.log("loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log("loss_wav", loss_wav, on_epoch=True, prog_bar=True)
        self.log("loss_stft", loss_stft, on_epoch=True, prog_bar=True)
        self.log("loss_adv", loss_adv, on_epoch=True, prog_bar=True)
        self.log("loss_feat", loss_feat, on_epoch=True, prog_bar=True)

        optimizer_g.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)

        ## For discriminator
        self.toggle_optimizer(optimizer_d)
        _, _, _, _, _, enhanced, clean,= self.forward(clean, ifloss=False)
        esti_fmap = self.model_d(enhanced.detach())
        true_fmap = self.model_d(clean)

        loss_dis = self.loss_dis(esti_fmap, true_fmap)

        optimizer_d.zero_grad()
        self.manual_backward(loss_dis)
        torch.nn.utils.clip_grad_norm_(self.model_d.parameters(), self.clip_grad_norm_value)
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)
        
        # _ = self.scheduler.step()
        # _ = self.scheduler_d.step()
        
        self.log("loss_dis", loss_dis, on_epoch=True, on_step=True, prog_bar=True)
        # self.log("lr", str(optimizer_g.param_groups[0]['lr']), on_epoch=True, on_step=True, prog_bar=True)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        # generator
        loss, loss_wav, loss_stft, loss_adv, loss_feat, enhanced, clean,= self.forward(batch)
        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log("val_loss_wav", loss_wav, on_epoch=True, on_step=True, prog_bar=True)
        self.log("val_loss_stft", loss_stft, on_epoch=True, on_step=True, prog_bar=True)
        self.log("val_loss_adv", loss_adv, on_epoch=True, on_step=True, prog_bar=True)
        self.log("val_loss_feat", loss_feat, on_epoch=True, on_step=True, prog_bar=True)

        loss_lsd = LSDLoss(enhanced, clean).item()
        loss_lsd2 = LSD(enhanced, clean).item()
        self.log("val_loss_lsd", loss_lsd, on_epoch=True, on_step=True, prog_bar=True)
        self.log("val_loss_lsd2", loss_lsd2, on_epoch=True, on_step=True, prog_bar=True)

        # discriminator
        _, _, _, _, _, enhanced, clean,= self.forward(clean, ifloss=False)
        esti_fmap = self.model_d(enhanced)
        true_fmap = self.model_d(clean)
        loss_dis = self.loss_dis(esti_fmap, true_fmap)
        self.log("val_loss_dis", loss_dis, on_epoch=True, on_step=True, prog_bar=True)

    def train_dataloader(self):
        return self.training_dataloader
    
    def val_dataloader(self):
        return self.validation_dataloader
        

if __name__ == '__main__':
    pl.seed_everything(1234, workers=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--config', default='/home/nis/xinan.chen/projects/BWE/TFGridNet/cfg_bwetf.toml')

    args = parser.parse_args()
    config = toml.load(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = config['DDP']['device']
    args.world_size = len(config['DDP']['device'].split(',')) 

    run(config, args)

    hold_gpu_memory(extra_gb=1, hold_hours=36)
