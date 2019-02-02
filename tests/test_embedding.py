from mac.mac import LazyEmbedding
import nose.tools


def test_embedding_make():
    embedding = LazyEmbedding(5, 32)
    res = embedding(['who dat boi', 'hello everyone'])
    nose.tools.assert_equal(res.shape,  (2, 3, 32))


def test_embedding_ovf():
    embedding = LazyEmbedding(5, 32)
    with nose.tools.assert_raises(IndexError):
        embedding(['who dat boi',
                   'hello everyone',
                   'mlem you are a beast'])
