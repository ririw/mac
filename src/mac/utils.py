from mac.config import getconfig


def cuda_message():
    if getconfig()['use_cuda']:
        print('CUDA enabled')
    else:
        print('CUDA disabled, this may be very slow...')
