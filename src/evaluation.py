import numpy as np
from models import generator
from natsort import natsorted
import os
# from tools.compute_metrics import compute_metrics
from utils import *
import torchaudio
import soundfile as sf
import argparse
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchaudio.functional as aF
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_avail", type=str, default="6", help="available GPUs")
parser.add_argument("--model_path", type=str, default='./src/best_ckpt/ckpt_04_30_12_58',
                    help="the path where the model is saved")
parser.add_argument("--test_dir", type=str, default='/data/hdd1/xinan.chen/VCTK_wav_single/test',
                    help="noisy tracks dir to be enhanced")
parser.add_argument("--sample_num", type=int, default=1, help="number of tracks to be saved and plotted")

args = parser.parse_args()

def plot_spectrogram(waveform, save_dir, title=None, n_fft=2048, hop_length=512):
    fig, ax = plt.subplots(figsize=(6, 4))
    spec = torch.stft(
        waveform, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        window=torch.hann_window(n_fft),
        return_complex=True
    )
    mag = torch.abs(spec)
    spec_dB = 20*torch.log10(mag.clamp(1e-8))
    # spec_dB [B, F, TT]
    assert spec_dB.shape[0] == 1
    spec_dB = spec_dB.squeeze(0)

    im = ax.imshow(spec_dB.cpu().numpy(), aspect='auto',origin='lower',vmin=-80, vmax=40)
    plt.colorbar(im, ax=ax)
    plt.xlabel('Frames')
    plt.ylabel('freq_bin')
    if title:
        plt.title(title)

    # save image
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, title + '.png'))


def cal_snr(pred, target):
    return (20 * torch.log10(torch.norm(target, dim=-1) / \
                                torch.norm(pred - target, dim=-1).clamp(min=1e-8))).mean()

def STFT(x, n_fft=2048, hop_length=512):
    stft = torch.stft(x,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        window=torch.hann_window(n_fft),
                        return_complex=False,
                        )  #[B, F, TT,2]
    mag = torch.norm(stft, p=2, dim =-1) #[B, F, TT]
    return mag

def cal_lsd_log10(pred, target):
    sp = torch.log10(STFT(pred).square().clamp(1e-8))
    st = torch.log10(STFT(target).square().clamp(1e-8))
    return (sp - st).square().mean(dim=1).sqrt().mean()

def cal_lsd_loge(pred, target):
    sp = torch.log(STFT(pred).square().clamp(1e-8))
    st = torch.log(STFT(target).square().clamp(1e-8))
    return (sp - st).square().mean(dim=1).sqrt().mean()


@torch.no_grad()
def enhance_one_track(
    model, noisy, cut_len, n_fft=400, hop=100
):
    noisy = noisy.cuda()

    # normalize
    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
    noisy = torch.transpose(noisy, 0, 1)
    noisy = torch.transpose(noisy * c, 0, 1)

    length = noisy.size(-1)
    frame_num = int(np.ceil(length / 100)) # ceil 向上取整
    padded_len = frame_num * 100
    padding_len = padded_len - length
    noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1) # 长度不足frame整数倍的部分用前面的数据补齐
    # if padded_len > cut_len: # 如果大于16s
    #     batch_size = int(np.ceil(padded_len / cut_len))
    #     # 莫名其妙？？？
    #     while 100 % batch_size != 0:
    #         batch_size += 1
    #     noisy = torch.reshape(noisy, (batch_size, -1))

    noisy_spec = torch.stft(
        noisy, n_fft, hop, window=torch.hamming_window(n_fft).cuda(), onesided=True,
        return_complex=True,
    )
    noisy_spec=torch.view_as_real(noisy_spec)
    # -> [B, F, T, 2]
    noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
    # -> [B, 2, T, F]
    est_real, est_imag = model(noisy_spec)
    # -> [batchsize,1,T,F]
    est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
    # -> [batchsize,1,F,T]

    est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
    # -> (batchsize,F,T,2)
    est_spec_uncompress=torch.view_as_complex(est_spec_uncompress)
    est_audio = torch.istft(
        est_spec_uncompress,
        n_fft,
        hop,
        window=torch.hamming_window(n_fft).cuda(),
        onesided=True,
    )
    # -> (batchsize,t)
    est_audio = est_audio / c # c为标量 反归一化 （无法处理多通道）
    # 长度还原
    est_audio = torch.flatten(est_audio)[:length]  # .cpu().numpy()
    # -> (t)
    assert len(est_audio) == length 

    return est_audio.unsqueeze(0).cpu(), length


def evaluation(args, noisy_dir, clean_dir):
    model_path = args.model_path
    sample_num = args.sample_num

    n_fft = 400
    model = generator.TSCNet(num_channel=64, num_features=n_fft // 2 + 1).cuda()
    model.load_state_dict((torch.load(model_path)))
    model.eval()

    audio_list = os.listdir(clean_dir)
    audio_list = natsorted(audio_list)
    num = len(audio_list)
    print("Total {} tracks to be enhanced and evaluated".format(num))
    # metrics_total = np.zeros(6)

    snr_list = []
    base_snr_list = []
    lsd_log10_list = []
    base_lsd_log10_list = []
    lsd_loge_list = []
    base_lsd_loge_list = []

    now_time = datetime.datetime.now().strftime('%m_%d_%H_%M')
    sample_saved_dir=os.path.join("./AudioSamples", now_time)
    if not os.path.exists(sample_saved_dir) and sample_num>0:
        os.makedirs(sample_saved_dir)

    for idx, audio in tqdm(enumerate(audio_list), total=num):
        noisy_path = os.path.join(noisy_dir, audio)
        clean_path = os.path.join(clean_dir, audio)
        clean_audio, sr1 = torchaudio.load(clean_path)
        noisy_audio, sr2 = torchaudio.load(noisy_path)
        length = clean_audio.size(-1)
        # noisy_audio =aF.resample(clean_audio, orig_freq=16000, new_freq=8000)
        # noisy_audio =aF.resample(noisy_audio, orig_freq=8000, new_freq=16000)
        noisy_audio = noisy_audio[:, : length]
        est_audio, _ = enhance_one_track(
            model, noisy_audio, 16000 * 16, n_fft, n_fft // 4
        )
        # clean_audio, sr = sf.read(clean_path)
        # noisy_audio, sr = sf.read(noisy_path)
        assert sr1 == 16000
        # metrics = compute_metrics(clean_audio, est_audio, sr, 0)

        snr_list.append(cal_snr(est_audio, clean_audio))
        base_snr_list.append(cal_snr(noisy_audio, clean_audio))
        lsd_log10_list.append(cal_lsd_log10(est_audio, clean_audio))
        base_lsd_log10_list.append(cal_lsd_log10(noisy_audio, clean_audio))
        lsd_loge_list.append(cal_lsd_loge(est_audio, clean_audio))
        base_lsd_loge_list.append(cal_lsd_loge(noisy_audio, clean_audio))

        # metrics = np.array(metrics)
        # metrics_total += metrics

        # 在给定数量的音频中，保存音频并绘制频谱

        if idx<sample_num:
            # 保存音频
            saved_path_est = os.path.join(sample_saved_dir, f'{audio[:-4]}_enhanced.wav')
            saved_path_clean = os.path.join(sample_saved_dir, f'{audio[:-4]}_clean.wav')
            saved_path_noisy = os.path.join(sample_saved_dir, f'{audio[:-4]}_noisy.wav')
            torchaudio.save(saved_path_est, est_audio, sr1)
            torchaudio.save(saved_path_clean, clean_audio, sr1)
            torchaudio.save(saved_path_noisy, noisy_audio, sr1)
 
            # 绘制频谱
            plot_spectrogram(est_audio, sample_saved_dir, title=f'{audio[:-4]}_enhanced')
            plot_spectrogram(clean_audio, sample_saved_dir, title=f'{audio[:-4]}_clean')
            plot_spectrogram(noisy_audio, sample_saved_dir, title=f'{audio[:-4]}_noisy')

    snr = torch.stack(snr_list, dim=0).mean()
    base_snr = torch.stack(base_snr_list, dim=0).mean()
    lsd_log10 = torch.stack(lsd_log10_list, dim=0).mean()
    base_lsd_log10 = torch.stack(base_lsd_log10_list, dim=0).mean()
    lsd_loge = torch.stack(lsd_loge_list, dim=0).mean()
    base_lsd_loge = torch.stack(base_lsd_loge_list, dim=0).mean()

    # metrics_avg = metrics_total / num
    print(
        "SNR: {:.4f} dB, LSD_log10: {:.4f}, LSD_loge: {:.4f}".format(
            snr, lsd_log10, lsd_loge
        )
        + "\n"
        + "base_SNR: {:.4f} dB, base_LSD_log10: {:.4f}, base_LSD_loge: {:.4f}".format(
            base_snr, base_lsd_log10, base_lsd_loge
        )
    )

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_avail
    noisy_dir = os.path.join(args.test_dir, "noisy")
    clean_dir = os.path.join(args.test_dir, "clean")
    evaluation(args, noisy_dir, clean_dir)
