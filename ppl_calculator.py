import torch
import metric_utils
import numpy as np
import copy
from model import Generator
from tqdm import tqdm
from torch import nn
from torchvision import  utils
import torch.nn.functional as F
import pdb
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
class GeneratorBlocks:
    def __init__(self,generator):
        self.G_Norm_MLP = list(generator.children())[0]
        self.G_Const_Input = list(generator.children())[1]
        self.conv1 = list(generator.children())[2]
        self.to_rgb1 = list(generator.children())[3]
        self.convs = list(generator.children())[4]
        self.to_rgbs = list(generator.children())[6]
        self.num_layers = 12
        self.noise = [None] * self.num_layers
    def G_Mapper(self,z_vector):
        styles = [self.G_Norm_MLP(s) for s in [z_vector]]
        # print(type(styles))
        latent = styles[0].unsqueeze(1).repeat(1, 12, 1)
        return styles,latent
    def G_Synthesis(self,latent):
        #### constant input block
        out = self.G_Const_Input(latent)
        # print("out shape is after G_const_Input :", out.shape)
        #### constant input block ends

        #### conv1 block
        out = self.conv1(out, latent[:, 0], noise=self.noise[0])
        # print("out shape is after conv1 :", out.shape)
        #### conv1 block ends

        #### to rgb1 block
        skip = self.to_rgb1(out, latent[:, 1])
        # print("out shape is after rgb1 :", out.shape)
        #### to rgb1 blokc ends

        #### remaining blocks
        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], self.noise[1::2], self.noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            # print("skip shape is : ", skip.shape)

            i += 2
        image = skip
        return image
    def saveImage(self,image,name):
        ####save image
        utils.save_image(
            image,
            name+".png",
            nrow=int(1 ** 0.5),
            normalize=True,
            range=(-1, 1),
        )
        ####save image ends
class PPLSampler(torch.nn.Module):
    def __init__(self, genBlock, epsilon, vgg16,space='w'):
        assert space in ['z', 'w']
        # assert sampling in ['full', 'end']
        super().__init__()
        self.genBlock = genBlock
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
            w0Vec,repeatw0Vec = self.genBlock.G_Mapper(z0)
            w1Vec,repeatw1Vec = self.genBlock.G_Mapper(z1)
            wt0 = torch.lerp(w0Vec[0], w1Vec[0], t)
            wt1 = torch.lerp(w0Vec[0], w1Vec[0], t + self.epsilon)
        else:  # space == 'z'
            zt0 = slerp(z0, z1, t.unsqueeze(1))
            zt1 = slerp(z0, z1, t.unsqueeze(1) + self.epsilon)
            wt0 = self.GMapper(zt0)
            wt1 = self.GMapper(zt1)

        #create 12 repeat vectors for wt0 and wt1 to send to generator
        wt0 = wt0.unsqueeze(1).repeat(1, 12, 1)
        wt1 = wt1.unsqueeze(1).repeat(1, 12, 1)

        # Generate images.
        img0 = self.genBlock.G_Synthesis(wt0)
        img1 = self.genBlock.G_Synthesis(wt1)

        # upscale
        img0 = F.interpolate(img0, 256, None, 'bilinear', True)
        img1 = F.interpolate(img1, 256, None, 'bilinear', True)
        pdb.set_trace()

        self.genBlock.saveImage(img0,"img0")
        self.genBlock.saveImage(img1,"img1")


        # Scale dynamic range from [-1,1] to [0,255].
        img0 = (img0 + 1) * (255 / 2)
        img1 = (img1 + 1) * (255 / 2)




        # Evaluate differential LPIPS.
        lpips_t0 = self.vgg16(img0, resize_images=False, return_lpips=True)
        lpips_t1 = self.vgg16(img1, resize_images=False, return_lpips=True)
        dist = (lpips_t0 - lpips_t1).square().sum(1) / self.epsilon ** 2
        pdb.set_trace()
        return dist

def compute_ppl(mainGenerator, num_samples, epsilon, batch_size, device):
    vgg16_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    vgg16 = metric_utils.get_feature_detector(vgg16_url, num_gpus=1, rank=0)

    genBlocks = GeneratorBlocks(mainGenerator)
    # Setup sampler.
    sampler = PPLSampler(genBlock=genBlocks, epsilon=epsilon, vgg16=vgg16)
    sampler.eval().requires_grad_(False).to(device)


    # Sampling loop.
    dist = []
    for batch_start in tqdm(range(0, num_samples, batch_size)):
        x = sampler(batch_size)
        # for src in range(1):
        #     y = x.clone()
        dist.append(x)

    # Compute PPL.
    dist = torch.cat(dist)[:num_samples].cpu().numpy()
    lo = np.percentile(dist, 1, interpolation='lower')
    hi = np.percentile(dist, 99, interpolation='higher')
    ppl = np.extract(np.logical_and(dist >= lo, dist <= hi), dist).mean()
    return float(ppl)

if __name__ == "__main__":
    with torch.no_grad():
        ckpt = "/common/users/sm2322/MS-Thesis/GAN-Thesis-Work-Remote/styleGAN2-AE-Ligong-Remote/trainedPts/gan/168000.pt"
        ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
        device = "cuda"
        generator = Generator(128, 512, 8, 2).to(device)
        generator.load_state_dict(ckpt["g_ema"])
        generator.eval()

        n_samples = 50000
        epsilon = 1e-4
        # epsilon=2e-1

        # print(list(generator.children()))
        # sample_z = torch.randn(1, 512, device=device)
        # genBlocks=GeneratorBlocks(generator)
        # wVecs,RepeatWVecs=genBlocks.G_Mapper(sample_z)
        # image=genBlocks.G_Synthesis(RepeatWVecs)
        # genBlocks.saveImage(image,"bullockCart")

        pplScore = compute_ppl(mainGenerator=generator, num_samples=n_samples, epsilon=epsilon,
                               batch_size=8, device=device)
        print("ppl score is :", pplScore)

