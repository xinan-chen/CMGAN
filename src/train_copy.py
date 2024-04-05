from models.generator import TSCNet, TSCNet_FRNN, TSCNet_DFRNN, TSCNet1
from models import discriminator
import os
from data import dataloader
import torch.nn.functional as F
import torch
from utils import power_compress, power_uncompress
import logging
from torchinfo import summary
import argparse
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=75, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--log_interval", type=int, default=0) # 多少个batch生成log
parser.add_argument("--decay_epoch", type=int, default=20, help="epoch from which to start lr decay")
parser.add_argument("--init_lr", type=float, default=5e-4, help="initial learning rate")
parser.add_argument("--cut_len", type=int, default=16000*2, help="cut length, default is 2 seconds in denoise "
                                                                 "and dereverberation")
parser.add_argument("--data_dir", type=str, default='/data/hdd1/xinan.chen/VCTK_wav_single_trim',
                    help="dir of VCTK+DEMAND dataset")
parser.add_argument("--save_model_dir", type=str, default='./saved_model',
                    help="dir of saved model")
parser.add_argument("--loss_weights", type=list, default=[0.0, 0.7, 0.0, 0.0],
                    help="weights of RI components, magnitude, time loss, and Metric Disc")

parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")

args = parser.parse_args()
logging.basicConfig(level=logging.INFO) # 将记录INFO级别及以上的日志


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    # 设置主节点的地址和端口号
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # 初始化进程组
    # 表示使用NCCL（NVIDIA Collective Communications Library）作为后端进行通信，它是NVIDIA提供的一个优化了GPU之间通信的库
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class Trainer:
    def __init__(self, train_ds, test_ds, gpu_id: int):

        self.n_fft = 400
        self.hop = 100
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.model = TSCNet1(num_channel=64, num_features=self.n_fft // 2 + 1).cuda()
        # 打印出模型的结构和参数信息。
        # summary(
        #     self.model, [(1, 2, args.cut_len // self.hop + 1, int(self.n_fft / 2) + 1)]
        # )
        # self.discriminator = discriminator.Discriminator(ndf=16).cuda()
        # summary(
        #     self.discriminator,
        #     [
        #         (1, 1, int(self.n_fft / 2) + 1, args.cut_len // self.hop + 1),
        #         (1, 1, int(self.n_fft / 2) + 1, args.cut_len // self.hop + 1),
        #     ],
        # )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.init_lr)
        # self.optimizer_disc = torch.optim.AdamW(
        #     self.discriminator.parameters(), lr=2 * args.init_lr
        # )
        self.model = DDP(self.model, device_ids=[gpu_id])
        # self.discriminator = DDP(self.discriminator, device_ids=[gpu_id])
        self.gpu_id = gpu_id

    def forward_generator_step(self, clean, noisy):

        # Normalization
        # 使数据标准差为1
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        #  [T, batchsize] * [batchsize,] -> [T, batchsize]
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(
            clean * c, 0, 1
        )
        # 输入 (batchsize,t)
        noisy_spec = torch.stft(
            noisy,
            self.n_fft,
            self.hop, # 窗口的跳跃步长，也就是相邻窗口之间的样本数
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True, # onesided=True 表示只计算并返回正频率。
            return_complex=True,
        )
        noisy_spec=torch.view_as_real(noisy_spec)

        clean_spec = torch.stft(
            clean,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
            return_complex=True,
        )
        clean_spec=torch.view_as_real(clean_spec)

        # -> (batchsize,F,T,2)

        # 将复数的幅度压缩为0，3次方
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        # -> (batchsize,2,T,F)
        clean_spec = power_compress(clean_spec)
        # -> (batchsize,2,F,T)
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)
        # -> (batchsize,1,F,T)

        est_real, est_imag = self.model(noisy_spec)
        # -> (batchsize,1,T,F)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        # -> (batchsize,1,F,T)
        est_mag = torch.sqrt(est_real**2 + est_imag**2)
        clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)
        # -> (batchsize,1,F,T)
        # 将复数的幅度还原
        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        # -> (batchsize,F,T,2)
        est_spec_uncompress=torch.view_as_complex(est_spec_uncompress)
        est_audio = torch.istft(
            est_spec_uncompress,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
        )
        # -> (batchsize,t)

        # return 除audio都为(batchsize,1,F,T)
        return {
            "est_real": est_real,
            "est_imag": est_imag,
            "est_mag": est_mag,
            "clean_real": clean_real,
            "clean_imag": clean_imag,
            "clean_mag": clean_mag,
            "est_audio": est_audio,
            "clean": clean,
        }

    def calculate_generator_loss(self, generator_outputs):

        # predict_fake_metric = self.discriminator(
        #     generator_outputs["clean_mag"], generator_outputs["est_mag"]
        # )
        # -> [batch,]
        # gen_loss_GAN = F.mse_loss(
        #     predict_fake_metric.flatten(), generator_outputs["one_labels"].float() # ？
        # )

        loss_mag = F.mse_loss(
            generator_outputs["est_mag"], generator_outputs["clean_mag"]
        )
        loss_ri = F.mse_loss(
            generator_outputs["est_real"], generator_outputs["clean_real"]
        ) + F.mse_loss(generator_outputs["est_imag"], generator_outputs["clean_imag"])

        time_loss = torch.mean(
            torch.abs(generator_outputs["est_audio"] - generator_outputs["clean"])
        )
  
        loss = (
            args.loss_weights[0] * loss_ri
            + args.loss_weights[1] * loss_mag
            + args.loss_weights[2] * time_loss
        )

        return loss

    def calculate_discriminator_loss(self, generator_outputs):

    #     length = generator_outputs["est_audio"].size(-1)
    #     est_audio_list = list(generator_outputs["est_audio"].detach().cpu().numpy())  # 不要传播generator的结果的梯度
    #     clean_audio_list = list(generator_outputs["clean"].cpu().numpy()[:, :length])
    #     # -> list：batch个np[length,]组成list
    #     pesq_score = discriminator.batch_pesq(clean_audio_list, est_audio_list)
    #     # -> [batch,]

    #     # The calculation of PESQ can be None due to silent part
    #     if pesq_score is not None:
    #         predict_enhance_metric = self.discriminator(
    #             generator_outputs["clean_mag"], generator_outputs["est_mag"].detach()
    #         ) # 不要传播generator的结果的梯度
    #         predict_max_metric = self.discriminator(
    #             generator_outputs["clean_mag"], generator_outputs["clean_mag"]
    #         )
    #         # .flatten() 是一个张量方法，用于将多维张量转换为一维张量
    #         discrim_loss_metric = F.mse_loss(
    #             predict_max_metric.flatten(), generator_outputs["one_labels"]
    #         ) + F.mse_loss(predict_enhance_metric.flatten(), pesq_score)
    #     else:
    #         discrim_loss_metric = None

    #     return discrim_loss_metric
        return None

    def train_step(self, batch):

        # Trainer generator
        clean = batch[0].to(self.gpu_id)
        noisy = batch[1].to(self.gpu_id)
        one_labels = torch.ones(args.batch_size).to(self.gpu_id)

        generator_outputs = self.forward_generator_step(
            clean,
            noisy,
        )
        generator_outputs["one_labels"] = one_labels
        # generator_outputs["clean"] = clean

        loss = self.calculate_generator_loss(generator_outputs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Train Discriminator
        discrim_loss_metric = self.calculate_discriminator_loss(generator_outputs)

        if discrim_loss_metric is not None:
            self.optimizer_disc.zero_grad()
            discrim_loss_metric.backward()
            self.optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.0])

        return loss.item(), discrim_loss_metric.item()

    @torch.no_grad()
    def test_step(self, batch):

        clean = batch[0].to(self.gpu_id)
        noisy = batch[1].to(self.gpu_id)
        one_labels = torch.ones(args.batch_size).to(self.gpu_id)

        generator_outputs = self.forward_generator_step(
            clean,
            noisy,
        )
        generator_outputs["one_labels"] = one_labels
        # generator_outputs["clean"] = clean

        loss = self.calculate_generator_loss(generator_outputs)

        discrim_loss_metric = self.calculate_discriminator_loss(generator_outputs)
        if discrim_loss_metric is None:
            discrim_loss_metric = torch.tensor([0.0])

        return loss.item(), discrim_loss_metric.item()

    def test(self):
        self.model.eval()
        # self.discriminator.eval()
        gen_loss_total = 0.0
        disc_loss_total = 0.0
        for idx, batch in enumerate(self.test_ds):
            step = idx + 1
            loss, disc_loss = self.test_step(batch)
            gen_loss_total += loss
            disc_loss_total += disc_loss
        gen_loss_avg = gen_loss_total / step
        disc_loss_avg = disc_loss_total / step

        # template = "GPU: {}, Generator loss: {}, Discriminator loss: {}"
        # logging.info(template.format(self.gpu_id, gen_loss_avg, disc_loss_avg))

        return gen_loss_avg, disc_loss_avg

    def train(self):
        # 它会在每个指定的步长 (step_size) 后，将当前的学习率乘以给定的因子 (gamma)。这种策略通常被称为 "步长衰减"。
        scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.decay_epoch, gamma=0.5
        )
        # scheduler_D = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer_disc, step_size=args.decay_epoch, gamma=0.5
        # )
        best_gen_loss = float(1000000)

        now_time = datetime.datetime.now().strftime('%m_%d_%H_%M')
        # save_model_dir = os.path.join(args.save_model_dir, now_time)

        if self.gpu_id == 0:
            writer = SummaryWriter(log_dir=os.path.join("runs", now_time))    
           
        for epoch in range(args.epochs):
            if self.gpu_id == 0:
                print("Epoch start:", epoch, now_time)
            self.model.train()
            # self.discriminator.train()
            gen_loss_total = 0.0
            disc_loss_total = 0.0
            for idx, batch in enumerate(self.train_ds):
                step = idx + 1
                loss, disc_loss = self.train_step(batch)
                gen_loss_total += loss
                disc_loss_total += disc_loss
                # template = "GPU: {}, Epoch {}, Step {}, loss: {}, disc_loss: {}"
                # if (step % args.log_interval) == 0:
                # if step == 1:
                #     logging.info(
                #         template.format(self.gpu_id, epoch, step, loss, disc_loss)
                #     )
            gen_loss_train = gen_loss_total / step
            disc_loss_train= disc_loss_total / step
            gen_loss_test, disc_loss_test= self.test() # 只用生成器loss

            # 保存模型

            # path = os.path.join(
            #     save_model_dir,
            #     "CMGAN_epoch_" + str(epoch) + "_" + str(gen_loss_test)[:5],
            # )

            if self.gpu_id == 0:
                writer.add_scalar('gen_loss_train',gen_loss_train , epoch) 
                writer.add_scalar('disc_loss_train',disc_loss_train , epoch)
                writer.add_scalar('gen_loss_test',gen_loss_test , epoch)
                writer.add_scalar('disc_loss_test',disc_loss_test , epoch)
                if gen_loss_test < best_gen_loss:
                    # torch.save(self.model.module.state_dict(), path)
                    best_model = self.model.module.state_dict()
                    best_gen_loss = gen_loss_test
                    best_epoch = epoch
                    writer.add_scalar('best_loss', gen_loss_test, epoch)
                    # best_path = path
                print("Epoch end:", epoch)
            # （可能）更新学习率
            scheduler_G.step() 
            # scheduler_D.step()
            
            
        # 将最好的模型复制到best_ckpt文件夹下
        if self.gpu_id == 0:
            logging.info("Best epoch: {}, Best loss: {}".format(best_epoch, best_gen_loss))
            torch.save(best_model, "./src/best_ckpt/ckpt_"+now_time)
            # os.system("cp " + best_path + " " + "./src/best_ckpt/ckpt_"+now_time) 
            # os.system("rm -rf " + args.save_model_dir)
            # 画图
            # plt.plot(gen_loss_list)
            # plt.xlabel('epoch')
            # plt.ylabel('loss')
            # plt.savefig(os.path.join("./src/best_ckpt", now_time + '.png'))
            writer.close()
        


def main(rank: int, world_size: int, args):
    import warnings
    warnings.filterwarnings('ignore')
    ddp_setup(rank, world_size)
    torch.cuda.set_device(rank)
    if rank == 0:
        print(args)
        available_gpus = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        print(available_gpus)

        # now_time = datetime.datetime.now().strftime('%m_%d_%H_%M')
        # save_model_dir = os.path.join(args.save_model_dir, now_time)
        # if not os.path.exists(save_model_dir):
        #     os.makedirs(save_model_dir)

        
    train_ds, test_ds = dataloader.load_data(
        args.data_dir, args.batch_size, args.n_cpu, args.cut_len
    )
    trainer = Trainer(train_ds, test_ds, rank)
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    # world_size = torch.cuda.device_count()
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
