import torch
from torch import nn
from torchvision import  utils
from model import Generator

with torch.no_grad():
    ckpt = "/common/users/sm2322/MS-Thesis/GAN-Thesis-Work-Remote/styleGAN2-AE-Ligong-Remote/trainedPts/aegan/168000.pt"
    ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
    device = "cuda"
    generator = Generator(128, 512, 8, 2).to(device)
    generator.load_state_dict(ckpt["g_ema"])
    generator.eval()
    # print(generator)

    #take the norm mlp block
    G_Norm_MLP = list(generator.children())[0]
    print(G_Norm_MLP)

    sample_z = torch.randn(1, 512, device=device)

    print("initial styles type is: ", type(sample_z))
    styles = [G_Norm_MLP(s) for s in [sample_z]]
    print(type(styles))
    latent = styles[0].unsqueeze(1).repeat(1, 12, 1)
    print(latent.shape)  # latent is wat the enters input block and moves forward



#save image
# img = G_Synthesis([sample_z])[0]
# print(img.shape)
# utils.save_image(
#     img,
#     "sampleTrainedImg.png",
#     nrow=int(1 ** 0.5),
#     normalize=True,
#     range=(-1, 1),
# )