import torch
import numpy as np
import tensorflow as tf


class SplitComb():
    def __init__(self, side_len, max_stride, stride, margin, pad_value):
        self.side_len = side_len
        self.max_stride = max_stride
        self.stride = stride
        self.margin = margin
        self.pad_value = pad_value

    def split(self, data, side_len=None, max_stride=None, margin=None):
        if side_len == None:
            side_len = self.side_len
        if max_stride == None:
            max_stride = self.max_stride
        if margin == None:
            margin = self.margin

        assert (side_len > margin)
        assert (side_len % max_stride == 0)
        assert (margin % max_stride == 0)

        splits = []
        _, z, h, w = data.shape

        nz = int(np.ceil(float(z) / side_len))
        nh = int(np.ceil(float(h) / side_len))
        nw = int(np.ceil(float(w) / side_len))

        nzhw = [nz, nh, nw]
        self.nzhw = nzhw

        pad = [[0, 0], [margin, nz * side_len - z + margin],
               [margin, nh * side_len - h + margin],
               [margin, nw * side_len - w + margin]]
        data = np.pad(data, pad, 'edge')

        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len + 2 * margin
                    sh = ih * side_len
                    eh = (ih + 1) * side_len + 2 * margin
                    sw = iw * side_len
                    ew = (iw + 1) * side_len + 2 * margin

                    split = data[np.newaxis, :, sz:ez, sh:eh, sw:ew]
                    splits.append(split)

        splits = np.concatenate(splits, 0)
        return splits, nzhw

    def combine(self,
                output,
                nzhw=None,
                side_len=None,
                stride=None,
                margin=None):

        if side_len == None:
            side_len = self.side_len
        if stride == None:
            stride = self.stride
        if margin == None:
            margin = self.margin
        if nzhw == None:
            nz = self.nz
            nh = self.nh
            nw = self.nw
        else:
            nz, nh, nw = nzhw
        assert (side_len % stride == 0)
        assert (margin % stride == 0)
        side_len /= stride
        margin /= stride

        splits = []
        for i in range(len(output)):
            splits.append(output[i])

        output = -1000000 * np.ones(
            (nz * side_len, nh * side_len, nw * side_len, splits[0].shape[3],
             splits[0].shape[4]), np.float32)

        idx = 0
        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len
                    sh = ih * side_len
                    eh = (ih + 1) * side_len
                    sw = iw * side_len
                    ew = (iw + 1) * side_len

                    split = splits[idx][margin:margin +
                                        side_len, margin:margin +
                                        side_len, margin:margin + side_len]
                    output[sz:ez, sh:eh, sw:ew] = split
                    idx += 1

        return output


def cs_block(self, x, stages=4, name=''):
    with tf.variable_scope(name):
        shortcut = x
        C = int(x.shape[-1]) // 2
        S = tf.spilt(x, 2, axis=-1)
        for i in range(stages):
            mean = (S[0] + S[1]) / 2.0
            S[0] = self.conv2d(S[0], C, self.Ksize, act='relu')
            S[0] = self.conv2d(S[0], C, self.Ksize)

            g = self.conv2d(S[1], 64, self.Ksize, act='relu')
            g = tf.concat([g, S[1]])
            S[1] = self.conv2d(g, C, self.Ksize)

            S[0] = S[0] + mean
            S[1] = s[1] + mean

        x = tf.concat(S, axis=-1)
        x = self.conv2d(x, C, 1)

        return shortcut + 0.1 * x
