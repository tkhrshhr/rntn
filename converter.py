import numpy as np

from chainer import cuda


def convert(batch, device):
    if device is None:
        def to_device(x):
            return x
    elif device < 0:
        to_device = cuda.to_cpu
    else:
        def to_device(x):
            return cuda.to_gpu(x, device, cuda.Stream.null)

    s_left = []
    s_right = []
    s_dest = []
    s_word = []

    o_left = []
    o_right = []
    o_dest = []
    o_word = []
    r_li = []
    for s_di, o_di, r in batch:
        s_left.append(s_di['lefts'])
        s_right.append(s_di['rights'])
        s_dest.append(s_di['dests'])
        s_word.append(s_di['words'])

        o_left.append(o_di['lefts'])
        o_right.append(o_di['rights'])
        o_dest.append(o_di['dests'])
        o_word.append(o_di['words'])

        r_li.append(r)

    return tuple([s_left + s_right + s_dest + s_word,
                 o_left + o_right + o_dest + o_word,
                 to_device(np.array(r_li, dtype=np.int32))])


def convert_relational(batch, device):
    if device is None:
        def to_device(x):
            return x
    elif device < 0:
        to_device = cuda.to_cpu
    else:
        def to_device(x):
            return cuda.to_gpu(x, device, cuda.Stream.null)

    s_left = []
    s_right = []
    s_dest = []
    s_ope = []
    s_word = []

    o_left = []
    o_right = []
    o_dest = []
    o_ope = []
    o_word = []

    r_li = []
    for s_di, o_di, r in batch:
        s_left.append(s_di['lefts'])
        s_right.append(s_di['rights'])
        s_dest.append(s_di['dests'])
        s_ope.append(s_di['opes'])
        s_word.append(s_di['words'])

        o_left.append(o_di['lefts'])
        o_right.append(o_di['rights'])
        o_dest.append(o_di['dests'])
        o_ope.append(o_di['opes'])
        o_word.append(o_di['words'])

        r_li.append(r)

    return tuple([s_left + s_right + s_dest + s_ope + s_word,
                 o_left + o_right + o_dest + o_ope + o_word,
                 to_device(np.array(r_li, dtype=np.int32))])
