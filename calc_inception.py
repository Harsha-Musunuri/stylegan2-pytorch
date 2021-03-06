import argparse
import pickle
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torchvision.models import inception_v3, Inception3
import numpy as np
from tqdm import tqdm
from PIL import Image

from utils_bigGAN import *
from inceptionBigGan import inceptionBigGAN_v3
from inception import InceptionV3
from dataset import MultiResolutionDataset, VideoFolderDataset, get_image_dataset


class Inception3Feature(Inception3):
    def forward(self, x):
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=True)

        x = self.Conv2d_1a_3x3(x)  # 299 x 299 x 3
        x = self.Conv2d_2a_3x3(x)  # 149 x 149 x 32
        x = self.Conv2d_2b_3x3(x)  # 147 x 147 x 32
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 147 x 147 x 64

        x = self.Conv2d_3b_1x1(x)  # 73 x 73 x 64
        x = self.Conv2d_4a_3x3(x)  # 73 x 73 x 80
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 71 x 71 x 192

        x = self.Mixed_5b(x)  # 35 x 35 x 192
        x = self.Mixed_5c(x)  # 35 x 35 x 256
        x = self.Mixed_5d(x)  # 35 x 35 x 288

        x = self.Mixed_6a(x)  # 35 x 35 x 288
        x = self.Mixed_6b(x)  # 17 x 17 x 768
        x = self.Mixed_6c(x)  # 17 x 17 x 768
        x = self.Mixed_6d(x)  # 17 x 17 x 768
        x = self.Mixed_6e(x)  # 17 x 17 x 768

        x = self.Mixed_7a(x)  # 17 x 17 x 768
        x = self.Mixed_7b(x)  # 8 x 8 x 1280
        x = self.Mixed_7c(x)  # 8 x 8 x 2048

        x = F.avg_pool2d(x, kernel_size=8)  # 8 x 8 x 2048

        return x.view(x.shape[0], x.shape[1])  # 1 x 1 x 2048


def load_patched_inception_v3():
    # inception = inception_v3(pretrained=True)
    # inception_feat = Inception3Feature()
    # inception_feat.load_state_dict(inception.state_dict())
    inception_feat = InceptionV3([3], normalize_input=False)

    return inception_feat


@torch.no_grad()
def extract_features(loader, inception, device):
    pbar = tqdm(loader)

    feature_list = []

    for img in pbar:
        if isinstance(img, (list, tuple)):  # (image, label) pair
            img = img[0]
        img = img.to(device)
        feature = inception(img)[0].view(img.shape[0], -1)
        feature_list.append(feature.to("cpu"))

    features = torch.cat(feature_list, 0)

    return features


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(
        description="Calculate Inception v3 features for datasets"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="image sizes used for embedding calculation",
    )
    parser.add_argument(
        "--batch", default=64, type=int, help="batch size for inception networks"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=50000,
        help="number of samples used for embedding calculation",
    )
    parser.add_argument(
        "--flip", action="store_true", help="apply random flipping to real images"
    )
    parser.add_argument("--eval_type", type=str, default='train')
    parser.add_argument("--name", type=str, default=None, help="name of inception embedding file")
    parser.add_argument("--dataset", type=str, default='multires')
    parser.add_argument("--cache", type=str, default=None)
    # parser.add_argument("path", metavar="PATH", help="path to datset lmdb file")
    parser.add_argument("--path", type=str, default=None,help='path to dataset')
    parser.add_argument("--inceptionCkpt", type=str, default=None, help='path to inception checkpoint')
    parser.add_argument("--prints", action='store_true', help="shoud I print debug lines")
    parser.add_argument("--use_torch", action='store_true', help="shoud I use torch for FID calc")

    args = parser.parse_args()



    dset = None
    if args.dataset == 'multires':
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5 if args.flip else 0),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        dset = MultiResolutionDataset(args.path, transform=transform, resolution=args.size)
    elif args.dataset == 'videofolder':
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5 if args.flip else 0),
                transforms.Resize(args.size),  # Image.LANCZOS
                transforms.CenterCrop(args.size),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dset = VideoFolderDataset(args.path, transform, mode='image', cache=args.cache)
    elif args.dataset == 'imagefolder':
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5 if args.flip else 0),
                transforms.Resize(args.size, Image.LANCZOS),
                transforms.CenterCrop(args.size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dset = datasets.ImageFolder(args.path, transform=transform)
    elif args.dataset == 'vggface': #for VGGFaces
        dset = get_vggfaces_data( size=args.size,data_root=args.path)
    else:
        dset = get_image_dataset(args, args.dataset, args.path, train=args.eval_type=='train')

    # args.n_sample = min(args.n_sample, len(dset))
    indices = torch.randperm(len(dset))[:args.n_sample]
    dset = Subset(dset, indices)
    print("length of dataset: ",len(dset),"\n")
    loader = DataLoader(dset, batch_size=args.batch, num_workers=4, shuffle=True)


    if args.dataset=='vggface':
        print("loading inception model \n")
        inception_model = inceptionBigGAN_v3(num_classes=2000,init_weights=False,pretrained=False, transform_input=False)
        inception_model.load_state_dict(
            torch.load(args.inceptionCkpt))
        inception_model = WrapInception(inception_model.eval()).cuda()
        inception_model = nn.DataParallel(inception_model)
        print("inception model loaded without issues \n")
        sample = sample_data(loader)
        pool, logits, labels = accumulate_inception_activations(sample, inception_model,
                                                                num_inception_images=args.n_sample)

        if args.prints:
            print('Calculating means and covariances...')
        if args.use_torch:
            mean, cov = torch.mean(pool, 0), torch_cov(pool, rowvar=False)
        else:
            mean, cov = np.mean(pool.cpu().numpy(), axis=0), np.cov(pool.cpu().numpy(), rowvar=False)
        if args.prints:
            print('Covariances calculated\n')
            print(f"extracted {pool.shape[0]} features")
    else:
        inception = load_patched_inception_v3()
        inception = nn.DataParallel(inception).eval().to(device)
        features = extract_features(loader, inception, device).numpy()

        # features = features[: args.n_sample]

        print(f"extracted {features.shape[0]} features")

        mean = np.mean(features, 0)
        cov = np.cov(features, rowvar=False)

    name = args.name or os.path.splitext(os.path.basename(args.path))[0]

    with open(f"inception_{name}.pkl", "wb") as f:
        pickle.dump({"mean": mean, "cov": cov, "size": args.size, "path": args.path}, f)
