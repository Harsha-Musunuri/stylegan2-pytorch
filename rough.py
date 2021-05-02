
import torch
from torch import nn
from torchvision import  utils
from model import Generator
import copy




with torch.no_grad():
    ckpt = "/common/users/sm2322/MS-Thesis/GAN-Thesis-Work-Remote/styleGAN2-AE-Ligong-Remote/trainedPts/aegan/168000.pt"
    ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
    device = "cuda"
    generator = Generator(128, 512, 8, 2).to(device)
    generator.load_state_dict(ckpt["g_ema"])
    generator.eval()
    print(list(generator.children()))

    #### Norm+MLP block
    G_Norm_MLP = list(generator.children())[0]
    # print(G_Norm_MLP)

    sample_z = torch.randn(1, 512, device=device)

    print("initial styles type is: ", type(sample_z))
    # if not input_is_latent:  # if `style' is z, then get w = self.style(z)
    styles = [G_Norm_MLP(s) for s in [sample_z]]
    # print(type(styles))
    latent = styles[0].unsqueeze(1).repeat(1, 12, 1)
    print(latent.shape) #latent is wat the enters input block and moves forward
    #### Norm+MLP blocks end
    num_layers=12
    noise = [None] * num_layers
    #### constant input block
    G_Const_Input = list(generator.children())[1]
    out = G_Const_Input(latent)
    print("out shape is after G_const_Input :",out.shape)
    #### constant input block ends

    #### conv1 block
    conv1=list(generator.children())[2]
    out = conv1(out, latent[:, 0], noise=noise[0])
    print("out shape is after conv1 :",out.shape)
    #### conv1 block ends

    #### to rgb1 block
    to_rgb1=list(generator.children())[3]
    skip = to_rgb1(out, latent[:, 1])
    print("out shape is after rgb1 :",out.shape)
    #### to rgb1 blokc ends

    #### remaining blocks
    convs=list(generator.children())[4]
    # print("convs is : ",convs)
    to_rgbs=list(generator.children())[6]
    # print("rgbs is :",to_rgbs)

    i = 1
    for conv1, conv2, noise1, noise2, to_rgb in zip(
            convs[::2], convs[1::2], noise[1::2], noise[2::2], to_rgbs
    ):
        out = conv1(out, latent[:, i], noise=noise1)
        out = conv2(out, latent[:, i + 1], noise=noise2)
        skip = to_rgb(out, latent[:, i + 2], skip)
        print("skip shape is : ", skip.shape)

        i += 2
    image=skip
    #### remaining blocks ends

    ####save image
    utils.save_image(
        image,
        "blockWiseThings.png",
        nrow=int(1 ** 0.5),
        normalize=True,
        range=(-1, 1),
    )
    ####save image ends













