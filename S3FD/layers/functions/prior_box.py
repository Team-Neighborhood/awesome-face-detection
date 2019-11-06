#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import torch
from itertools import product as product
import math

from torch.autograd import Function


class PriorBox(Function):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    @staticmethod
    def forward(ctx, input_size, feature_maps, cfg):

        imh = input_size[0]
        imw = input_size[1]

        # number of priors for feature map location (either 4 or 6)
        variance = cfg.VARIANCE or [0.1]
        min_sizes = cfg.ANCHOR_SIZES
        steps = cfg.STEPS
        clip = cfg.CLIP
        for v in variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

        mean = []
        for k in range(len(feature_maps)):
            feath = feature_maps[k][0]
            featw = feature_maps[k][1]
            for i, j in product(range(feath), range(featw)):
                f_kw = imw / steps[k]
                f_kh = imh / steps[k]

                cx = (j + 0.5) / f_kw
                cy = (i + 0.5) / f_kh

                s_kw = min_sizes[k] / imw
                s_kh = min_sizes[k] / imh

                mean += [cx, cy, s_kw, s_kh]

        output = torch.Tensor(mean).view(-1, 4)
        if clip:
            output.clamp_(max=1, min=0)
        return output


if __name__ == '__main__':
    from data.config import cfg
    p = PriorBox([640, 640], cfg)
    out = p.forward()
    print(out.size())
