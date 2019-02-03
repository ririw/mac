import numpy as np
from fs import open_fs
from plumbum import cli

from mac import utils, datasets


@utils.MAC.subcommand('train')
class Train(cli.Application):
    def main(self, clevr_dir, preproc_dir, results_loc, log_loc):
        utils.cuda_message()
        np.printoptions(linewidth=139)

        clevr_fs = open_fs(clevr_dir, create=False)
        preproc_fs = open_fs(preproc_dir, create=True)

        dataset = datasets.TaskDataset(clevr_fs, preproc_fs, 'val')
        print(dataset[[3, 4, 5]])
