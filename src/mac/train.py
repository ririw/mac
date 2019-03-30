import datetime

import tensorboardX
import torch

import numpy as np
from fs import open_fs
from plumbum import cli
from torch.utils import data
from torch.nn import functional
from tqdm import tqdm

from mac import utils, datasets, mac, config


@utils.MAC.subcommand("train")
class Train(cli.Application):
    def main(self, clevr_dir, preproc_dir, results_loc, log_loc=None):
        utils.cuda_message()
        np.printoptions(linewidth=139)

        clevr_fs = open_fs(clevr_dir, create=False)
        preproc_fs = open_fs(preproc_dir, create=True)

        dataset = datasets.TaskDataset(clevr_fs, preproc_fs, "train")
        total_words = len(dataset.word_ix) + 1
        sampler = data.BatchSampler(data.SequentialSampler(dataset), 32, False)

        net = mac.MACNet(mac.MACRec(12, 512), total_words).to(config.torch_device())
        opt = torch.optim.Adam(net.parameters())

        if log_loc:
            now = datetime.datetime.now()
            log_dir = f"{log_loc}/new-{now}"
            writer = tensorboardX.SummaryWriter(log_dir)
        else:
            writer = None

        step = 0
        rolling_accuracy = 0
        for epoch in range(10):
            bar = tqdm(sampler)
            for batch_ix in bar:
                opt.zero_grad()
                images, qns, qn_lens, answers = dataset[batch_ix]
                predictions = net(images, qns, qn_lens)

                loss = functional.cross_entropy(predictions, answers)
                loss.backward()
                opt.step()
                hard_preds = np.argmax(predictions.detach().cpu().numpy(), 1)
                accuracy = (hard_preds == answers.detach().cpu().numpy()).mean()
                if writer is not None:
                    writer.add_scalar("loss", loss.item(), step)
                    writer.add_scalar("accuracy", accuracy, step)

                rolling_accuracy = rolling_accuracy * 0.95 + accuracy * 0.05
                bar.set_description("Accuracy: {}".format(rolling_accuracy))

                step += 1
