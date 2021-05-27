import torch
from torchvision import  utils
from model import Generator
import lpips
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
        print("out shape is after G_const_Input :", out.shape)
        #### constant input block ends

        #### conv1 block
        out = self.conv1(out, latent[:, 0], noise=self.noise[0])
        print("out shape is after conv1 :", out.shape)
        #### conv1 block ends

        #### to rgb1 block
        skip = self.to_rgb1(out, latent[:, 1])
        print("out shape is after rgb1 :", out.shape)
        #### to rgb1 blokc ends

        #### remaining blocks
        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], self.noise[1::2], self.noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            print("skip shape is : ", skip.shape)

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



if __name__ == "__main__":
    loss_fn_vgg = lpips.PerceptualLoss(net='vgg')
    with torch.no_grad():
        ckpt = "/common/users/sm2322/MS-Thesis/GAN-Thesis-Work-Remote/styleGAN2-AE-Ligong-Remote/trainedPts/gan/168000.pt"
        ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
        device = "cuda"
        generator = Generator(128, 512, 8, 2).to(device)
        generator.load_state_dict(ckpt["g_ema"])
        generator.eval()
        # print(list(generator.children()))
        z0, z1 = torch.randn([1 * 2, 512], device=device).chunk(2)
        # sample_z = torch.randn(1, 512, device=device)

        genBlocks=GeneratorBlocks(generator)
        w0,stackW0=genBlocks.G_Mapper(z0)
        w1,stackW1=genBlocks.G_Mapper(z1)

        epsilon = 1e-4
        t = torch.rand(1,device=device)[:, None]
        print(t.shape)
        interpolated_1 = torch.lerp(w0[0], w1[0], t)
        interpolated_2 = torch.lerp(w0[0], w1[0], t + epsilon)
        print(interpolated_1.shape, interpolated_2.shape)
        interpolated_1 = interpolated_1.unsqueeze(1).repeat(1, 12, 1)
        interpolated_2 = interpolated_2.unsqueeze(1).repeat(1, 12, 1)
        image1=genBlocks.G_Synthesis(interpolated_1)
        image2=genBlocks.G_Synthesis(interpolated_2)
        genBlocks.saveImage(image1,"image1_1-synth-latest")
        genBlocks.saveImage(image2,"image2_1-synth-latest")
        cur_lpips = loss_fn_vgg(image1, image2).item()
        print(f"Image LPIPS is {cur_lpips}")
        ppl = cur_lpips / (epsilon ** 2)
        print(f"Our final sample PPL is {ppl}")


        # generator([z0])