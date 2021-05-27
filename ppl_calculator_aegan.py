import pdb

import torch
import metric_utils
import numpy as np
import copy
from model import Generator,Encoder
from tqdm import tqdm

# print(generator)

sampling='full'
# Spherical interpolation of a batch of vectors.
def slerp(a, b, t):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    d = (a * b).sum(dim=-1, keepdim=True)
    p = t * torch.acos(d)
    c = b - d * a
    c = c / c.norm(dim=-1, keepdim=True)
    d = a * torch.cos(p) + c * torch.sin(p)
    d = d / d.norm(dim=-1, keepdim=True)
    return d
class PPLSampler(torch.nn.Module):
    def __init__(self, GMapper,Generator, epsilon, vgg16,space='w'):
        assert space in ['z', 'w']
        # assert sampling in ['full', 'end']
        super().__init__()
        self.GMapper = copy.deepcopy(GMapper)
        self.Generator = copy.deepcopy(Generator)
        self.epsilon = epsilon
        # self.sampling = sampling
        self.vgg16 = copy.deepcopy(vgg16)
        self.space=space

    def forward(self, batch_size):
        # Generate random latents and interpolation t-values.
        t = torch.rand(batch_size,device=device)[:, None]
        z0, z1 = torch.randn([batch_size * 2, 512], device=device).chunk(2)  # 2 pairs of n_sample number of points

        # Interpolate in W or Z.
        if self.space == 'w':
            w0 = self.GMapper(z0)
            w1 = self.GMapper(z1)
            wt0 = torch.lerp(w0, w1, t)
            wt1 = torch.lerp(w0, w1, t + self.epsilon)
        else:  # space == 'z'
            zt0 = slerp(z0, z1, t.unsqueeze(1))
            zt1 = slerp(z0, z1, t.unsqueeze(1) + self.epsilon)
            wt0 = self.GMapper(zt0)
            wt1 = self.GMapper(zt1)


        # Generate images.
        img0 = self.Generator([wt0])[0]
        img1 = self.Generator([wt1])[0]

        # Scale dynamic range from [-1,1] to [0,255].
        img0 = (img0 + 1) * (255 / 2)
        img1 = (img1 + 1) * (255 / 2)


        # Evaluate differential LPIPS.
        lpips_t0 = self.vgg16(img0, resize_images=False, return_lpips=True)
        lpips_t1 = self.vgg16(img1, resize_images=False, return_lpips=True)
        dist = (lpips_t0 - lpips_t1).square().sum(1) / self.epsilon ** 2
        return dist

def compute_ppl(Generator,GMapper, num_samples, epsilon, batch_size, device):
    vgg16_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    vgg16 = metric_utils.get_feature_detector(vgg16_url, num_gpus=1, rank=0)

    # Setup sampler.
    sampler = PPLSampler(GMapper=GMapper, Generator=Generator, epsilon=epsilon, vgg16=vgg16)
    sampler.eval().requires_grad_(False).to(device)


    # Sampling loop.
    dist = []
    for batch_start in tqdm(range(0, num_samples, batch_size)):
        x = sampler(batch_size)
        for src in range(1):
            y = x.clone()
            dist.append(y)

    # Compute PPL.
    dist = torch.cat(dist)[:num_samples].cpu().numpy()
    lo = np.percentile(dist, 1, interpolation='lower')
    hi = np.percentile(dist, 99, interpolation='higher')
    ppl = np.extract(np.logical_and(dist >= lo, dist <= hi), dist).mean()
    return float(ppl)


if __name__ == "__main__":
    with torch.no_grad():
        ckpt = "/common/users/sm2322/MS-Thesis/GAN-Thesis-Work-Remote/styleGAN2-AE-Ligong-Remote/trainedPts/aegan/168000.pt"
        ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
        pdb.set_trace()
        # device = "cuda"
        # generator = Generator(128, 512, 8, 2).to(device)
        # encoder = Encoder(128, 512, 2,which_latent='w_plus',
        #                 which_phi='lin2',
        #                 stddev_group=1).to(device)
        # generator.load_state_dict(ckpt["g_ema"])
        # encoder.load_state_dict(ckpt['e_ema'])
        # generator.eval()
        # encoder.eval()
        #
        # n_samples = 50000
        # epsilon = 1e-4
        #
        # # sample_gz, _ = g_ema([sample_z])
        # # latent_gz, _ = e_ema(sample_gz)
        # # rec_fake, _ = g_ema([latent_gz], input_is_latent=input_is_latent)
        #
        # for module in generator.named_children():
        #     GMapper = copy.deepcopy(module[1])
        #     break
        # pplScore=compute_ppl(Generator=generator,GMapper=GMapper,num_samples=n_samples,epsilon=epsilon,batch_size=2,device=device)
        # print("ppl score is :",pplScore)