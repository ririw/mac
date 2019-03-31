import numpy as np
import torch
from PIL import Image
from fs.base import FS
from torch.utils import data
from torchvision.models import resnet101
from torchvision.transforms import transforms
from tqdm import tqdm

from mac import config

transform = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.Pad(4),
        transforms.RandomCrop([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


class RawImageDataset(data.Dataset):
    def __init__(self, base_fs: FS, ds_name="train"):
        self.split = ds_name
        self.base_fs = base_fs.opendir("images/{}".format(ds_name))
        self.length = len(self.base_fs.listdir(""))

    def __getitem__(self, index):
        imname = "CLEVR_{}_{:06d}.png".format(self.split, index)
        with self.base_fs.open(imname, "rb") as img:
            img = Image.open(img).convert("RGB")
        return transform(img)

    def __len__(self):
        return self.length


def get_preprocess_images(base_fs, output_fs, split):
    fname = "{}-image".format(split)
    if not output_fs.exists(fname):
        _preprocess_images(base_fs, output_fs, split)

    result_path = output_fs.getsyspath(fname)
    res_flat = np.memmap(result_path, np.float32, "r")
    return res_flat.reshape(-1, 1024, 14, 14)


def _preprocess_images(base_fs, output_fs: FS, split):
    fname = "{}-image-tmp".format(split)
    result_path = output_fs.getsyspath(fname)

    batch_size = 32
    img_ds = RawImageDataset(base_fs, split)
    dataloader = data.DataLoader(img_ds, batch_size=batch_size, num_workers=2)
    result_size = (len(img_ds), 1024, 14, 14)
    result = np.memmap(result_path, np.float32, "w+", shape=result_size)

    resnet = get_resnet().to(config.torch_device())

    with torch.no_grad():
        progbar = tqdm(dataloader, desc="Preprocessing images -- {}".format(split))
        for ix, img in enumerate(progbar):
            img = resnet(img.to(config.torch_device())).cpu().numpy()
            result[ix * batch_size : (ix + 1) * batch_size] = img

    output_fs.move(fname, "{}-image".format(split), True)


def get_resnet():
    resnet = resnet101(True)
    preproc_net = torch.nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
    )
    return preproc_net
