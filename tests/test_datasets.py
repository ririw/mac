import fs.tempfs
import numpy as np
from mac import datasets
import nose.tools


def test_dataset_single():
    fake_ds, dataset_data = make_fake_dataset()
    with datasets.MAC_NP_Dataset(fake_ds, 'val') as val_dataset:
        for i in range(16):
            answer, question, image_ix, image = val_dataset[i]
            nose.tools.assert_equal(image_ix, dataset_data['val']['img_ix'][i])
            np.testing.assert_array_equal(
                image, dataset_data['val']['images'][image_ix])
            np.testing.assert_array_equal(
                question, dataset_data['val']['question'][i])
            np.testing.assert_array_equal(
                answer, dataset_data['val']['answer'][i])
            np.testing.assert_array_less(image_ix, 16)


def test_dataset_multiple():
    fake_ds, dataset_data = make_fake_dataset()
    with datasets.MAC_NP_Dataset(fake_ds, 'val') as val_dataset:
        for i in range(10):
            answer, question, image_ix, image = val_dataset[i:i + 6]
            np.testing.assert_array_equal(
                image_ix, dataset_data['val']['img_ix'][i:i + 6])
            np.testing.assert_array_equal(
                image, dataset_data['val']['images'][image_ix])
            np.testing.assert_array_equal(
                question, dataset_data['val']['question'][i:i + 6])
            np.testing.assert_array_equal(
                answer, dataset_data['val']['answer'][i:i + 6])

            nose.tools.assert_equal(len(image_ix), 6)
            np.testing.assert_array_less(image_ix, 16)


def test_dataset_fail_ctx_get():
    fake_ds, dataset_data = make_fake_dataset()
    ds = datasets.MAC_NP_Dataset(fake_ds, 'val')
    with nose.tools.assert_raises(TypeError):
        x = ds[0]
        assert x is not None  # Dummy test for pep8


def test_dataset_fail_ctx_len():
    fake_ds, dataset_data = make_fake_dataset()
    ds = datasets.MAC_NP_Dataset(fake_ds, 'val')
    with nose.tools.assert_raises(TypeError):
        assert len(ds)  # Dummy test for pep8


def make_fake_dataset():
    temp_fs = fs.tempfs.TempFS()

    batch_size = 16
    all_data = {}
    for group in {'train', 'val'}:
        sg = temp_fs.makedir(group)
        images = np.memmap(
            sg.getsyspath('images'), mode='w+',
            dtype=np.float32, shape=(batch_size, 1024, 14, 14))
        img_ix = np.memmap(
            sg.getsyspath('img_ix'), mode='w+',
            dtype=np.int32, shape=batch_size)
        # This ensures that some unit tests pass
        question = np.memmap(
            sg.getsyspath('question'), mode='w+',
            dtype=np.float32, shape=(batch_size, 160, 256))
        answer = np.memmap(
            sg.getsyspath('answer'), mode='w+',
            dtype=np.int32, shape=batch_size)

        images[:] = np.random.normal(size=images.shape)
        img_ix[:] = np.random.choice(batch_size, size=img_ix.shape)
        question[:] = np.random.normal(batch_size, size=question.shape)
        answer[:] = np.random.choice(28, size=answer.shape)
        img_ix[0] = 0

        all_data[group] = {
            'images': images,
            'img_ix': img_ix,
            'question': question,
            'answer': answer,
        }

    return temp_fs, all_data
