import numpy as np
import torch
import os
import datetime
import torchaudio
import matplotlib.pyplot as plt
from pesq import pesq  
from scipy.signal import iirfilter, filtfilt, firwin
import random
from tqdm import tqdm

from torchaudio import transforms
from torch.utils.tensorboard import SummaryWriter

def plot_spectrogram(waveform, save_dir, title=None, n_fft=2048, hop_length=512):
    fig, ax = plt.subplots(figsize=(6, 4))
    spec = torch.stft(
        waveform, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        window=torch.hamming_window(n_fft),
        return_complex=False
    )
    mag = torch.norm(spec, p=2, dim =-1)
    spec_dB = 20*torch.log10(mag.clamp(1e-8))
    # spec_dB [B, F, TT]
    assert spec_dB.shape[0] == 1
    spec_dB = spec_dB.squeeze(0)

    im = ax.imshow(spec_dB.cpu().numpy(), aspect='auto',origin='lower')
    plt.colorbar(im, ax=ax)
    plt.xlabel('Frames')
    plt.ylabel('freq_bin')
    if title:
        plt.title(title)

    # save image
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, title + '.png'))

if __name__ == '__main__':
    # print('Hello World!')
    # print(torch.cuda.device_count())
    # print(torch.cuda.get_device_name(0))

    # noisy=torch.arange(0, 12).reshape(3, 4)
    # c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
    # print(noisy.size(-1))
    # c = os.path.join("/data","train")
    
    # 获取当前时间
    # now_time = datetime.datetime.now().strftime('%m_%d_%H_%M')

    # 生成随机音频
    # waveform = torch.randn(1, 16000)
    # plot_spectrogram(waveform, save_dir=".",title='Waveform')
    # os.system("cp " + "./src/best_ckpt/ckpt" + " " + "ckpt_"+now_time)

    # test = torch.randn(1, 2074944).contiguous()
    # test = test.view(1, -1,201,64)
    # print(2074944/101/64)
    # world_size = torch.cuda.device_count()
    # print(world_size)
    # waveform, sample_rate = torchaudio.load("/data/hdd1/xinan.chen/VCTK_wav_single/train/clean/p225_001_mic1.wav", normalize=True)
    # transform = transforms.Vad(sample_rate=sample_rate)
    # waveform_start_trim = transform(waveform)
    # flipped_audio = torch.flip(waveform_start_trim , [1])
    # waveform_end_trim = transform(flipped_audio)
    # waveform_trim = torch.flip(waveform_end_trim, [1])
    

    # torchaudio.save("./Visualizations/1.wav", waveform_trim, sample_rate)
    # plot_spectrogram(waveform_trim, save_dir="./Visualizations",title='WaveformAfter')
    # plot_spectrogram(waveform, save_dir="./Visualizations",title='WaveformBefore')

    # writer = SummaryWriter()

    # # 使用add_scalar方法
    # for n_iter in range(100):
    #     writer.add_scalar('Loss/train', np.random.random(), n_iter)       # 自定义名称Loss/train,并可视化监控
    #     writer.add_scalar('Loss/test', np.random.random(), n_iter)
    #     writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    #     writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

        
    # # 使用add_image方法    
    # # 构建一个100*100，3通道的img数据    
    # img = np.zeros((3, 100, 100))
    # img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
    # img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

    # writer.add_image('my_image', img, 0)

    # writer.close()
    # cv = 0
    # for i in tqdm(range(1)):
    #     if cv == 0:
    #         # train
    #         ForI = random.randint(0, 1)
    #         if ForI == 0:
    #             # FIR
    #             numtaps = random.randint(7, 31)
    #             cutoff = 0.5
    #             win = random.choice(['hamming', 'hann', 'blackman', 'bartlett'])
    #             b=firwin(numtaps, cutoff, window = win, pass_zero='lowpass')
    #             a=1
    #         else:
    #             # IIR
    #             N = random.randint(4, 12)
    #             Wn = 0.5
    #             ft = random.choice(['butter', 'cheby1', 'cheby2', 'ellip', 'bessel'])
    #             if ft == 'cheby1':
    #                 rp = random.choice([1e-6, 1e-3, 1, 5])
    #                 rs = None
    #             elif ft == 'cheby2':
    #                 rp = None
    #                 rs = random.choice([20, 40, 60, 80])
    #             elif ft == 'ellip':
    #                 rp = random.choice([1e-6, 1e-3, 1, 5])
    #                 rs = random.choice([20, 40, 60, 80])
    #             else:
    #                 rp = None
    #                 rs = None
    #             b, a =iirfilter(N, Wn, rp=rp, rs=rs,btype='lowpass',ftype=ft, output='ba')
    #     else:
    #         # test
    #         order = 8
    #         rp = 0.05
    #         b, a =iirfilter(order, 0.5, rp, btype='lowpass', ftype='cheby1', output='ba')

    #     wav = torch.randn(16000)
    #     wav_l = filtfilt(b, a, wav.numpy())
    #     wav_l = torch.from_numpy(wav_l.copy())
    #     assert len(wav_l) == len(wav)
    # pass
    # p=3
    # a = torch.randn(4, 80)
    # length = a.size(1)
    # a = a[:,:length-length%p]
    # print(a.size(1))
    # print(torch.arange(1, a.size(1), p))
    # a = torch.randn(4, 80)
    # spec = torch.stft(a, n_fft=32, hop_length=8, window=torch.hamming_window(32), return_complex=True)
    # print(torch.abs(spec).size())
    
    # b= 2.5+3.4j
    # b = torch.tensor([b])
    # print(torch.abs(b))
    # p=[]
    # p.append(torch.ones(2,2))
    # p.append(torch.ones(2,2))
    # sum = sum(a for a in p)
    # print(sum)
    b= 1+2j
    b = torch.tensor([b])
    c = torch.tensor([b])*2
    from torch.nn.functional import mse_loss
    s = mse_loss(b, c)
    print(s)

