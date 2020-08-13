# -*- coding: utf-8 -*-
import numpy as np
import torch

"""
naive op(function in torch) using numpy to vertify the coinsidece
with the algorithm in pytorch
"""


def batchnorm(input_image, weight, bias, running_mean,
              running_var, eps=1e-5, momentum=0.1, training=True):
    r"""
    Args:
        input_image: shape(NCHW)
        weight: (C)
        bias:(C)
        running_mean: statistic mean of batch in total
        running_var: statistic var of batch in total
        eps: numerical stable
        momentum: average the running mean and running var
    Return:
        output_image:(NCHW)
    """
    if training:
        # update statistic
        current_sum = np.sum(input_image, axis=0).sum(axis=1).sum(axis=1)
        total_num = input_image.size/input_image.shape[1]
        current_mean = current_sum/total_num  # (C,)
        # reshape first
        current_mean_reshaped = current_mean.reshape(1, 3, 1, 1)
        current_var_reshaped = np.power(
            (input_image-current_mean_reshaped), 2)/(total_num)
        current_var = current_var_reshaped.sum(axis=0).sum(axis=1).sum(axis=1)
        running_mean = running_mean*(1-momentum)+current_mean*momentum
        running_var = running_var*(1-momentum)+current_var*momentum

    running_var_reshaped = running_var.reshape(1, 3, 1, 1)
    running_mean_reshaped = running_mean.reshape(1, 3, 1, 1)
    y = (input_image-running_mean_reshaped)/(np.sqrt(running_var_reshaped+eps))
    y = y*weight.reshape(1, 3, 1, 1)+bias.reshape(1, 3, 1, 1)

    return y


def main():
    precision = 1e-7
    # pth
    input_image_pth = torch.rand(1, 3, 224, 224)
    batchnorm_pth = torch.nn.BatchNorm2d(3)
    batchnorm_pth.eval()
    output_image_pth = batchnorm_pth(input_image_pth)
    weight = batchnorm_pth.weight.detach().numpy()
    bias = batchnorm_pth.bias.detach().numpy()
    running_mean = batchnorm_pth.running_mean.detach().numpy()
    running_var = batchnorm_pth.running_var.detach().numpy()
    eps = batchnorm_pth.eps
    momentum = batchnorm_pth.momentum

    # np
    input_image_np = input_image_pth.numpy()
    output_image_np = batchnorm(input_image_np, weight, bias, running_mean,
                                running_var, eps, momentum, training=False)

    # check the result
    # TODO(breakpoint) failed when training=True
    assert np.all(np.abs(output_image_pth.detach().numpy() -
                         output_image_np) < precision)
    print('BatchNorm2d Assert Success!')


if __name__ == '__main__':
    main()
