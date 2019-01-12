import h5py
import click
import fs
from tqdm import tqdm
import numpy as np


@click.command()
@click.argument('input_file_name')
@click.argument('output_file_name')
def decompress(input_file_name, output_file_name):
    input_file = h5py.File(input_file_name, 'r')
    output_file = h5py.File(output_file_name + '.h5py', mode='w')
    output_dir = fs.open_fs(output_file_name, create=True)

    for group_key in input_file:
        group = input_file[group_key]
        output_ft_group = output_file.create_group(group_key)
        output_group = output_dir.makedir(group_key, recreate=True)
        for ds_key in group:
            ds = group[ds_key]
            new_f5_ds = output_ft_group.create_dataset(
                ds_key, shape=ds.shape,
                dtype=ds.dtype, chunks=True,
                compression='lzf'
            )
            if ds.dtype == np.float32:
                dtype = 'float16'
            else:
                dtype = ds.dtype
            new_ds = np.memmap(
                output_group.getsyspath(ds_key),
                dtype=dtype,
                mode='w+',
                shape=ds.shape,
            )
            for i in tqdm(range(0, ds.shape[0], 200),
                          desc='{} -- {}'.format(group_key, ds_key)):
                new_ds[i:i+200] = ds[i:i+200]
                new_f5_ds[i:i+200] = ds[i:i+200]
    input_file.close()


if __name__ == '__main__':
    decompress()
