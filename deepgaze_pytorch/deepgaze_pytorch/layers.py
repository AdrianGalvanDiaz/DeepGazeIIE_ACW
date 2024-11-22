# layers.py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    __constants__ = ['features', 'weight', 'bias', 'eps', 'center', 'scale']
    def __init__(self, features, eps=1e-12, center=True, scale=True):
        super().__init__()
        self.features = features
        self.eps = eps
        self.center = center
        self.scale = scale

        if self.scale:
            self.weight = nn.Parameter(torch.Tensor(self.features))
        else:
            self.register_parameter('weight', None)

        if self.center:
            self.bias = nn.Parameter(torch.Tensor(self.features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.scale:
            nn.init.ones_(self.weight)
        if self.center:
            nn.init.zeros_(self.bias)

    def adjust_parameter(self, tensor, parameter):
        return torch.repeat_interleave(
            torch.repeat_interleave(
                parameter.view(-1, 1, 1),
                repeats=tensor.shape[2],
                dim=1),
            repeats=tensor.shape[3],
            dim=2
        )

    def forward(self, input):
        normalized_shape = (self.features, input.shape[2], input.shape[3])
        weight = self.adjust_parameter(input, self.weight)
        bias = self.adjust_parameter(input, self.bias)
        return F.layer_norm(input, normalized_shape, weight, bias, self.eps)
    
    def extra_repr(self):
        return '{features}, eps={eps}, ' \
            'center={center}, scale={scale}'.format(**self.__dict__)

# def gaussian_filter_1d(tensor, dim, sigma, truncate=4, kernel_size=None, padding_mode='replicate', padding_value=0.0):
#     sigma = torch.as_tensor(sigma, device=tensor.device, dtype=tensor.dtype)
    
#     if kernel_size is not None:
#         kernel_size = torch.as_tensor(kernel_size, device=tensor.device, dtype=torch.int64)
#     else:
#         kernel_size = torch.as_tensor(2 * torch.ceil(truncate * sigma) + 1, device=tensor.device, dtype=torch.int64)
    
#     kernel_size = kernel_size.detach()
#     kernel_size_int = kernel_size.detach().cpu().numpy()
    
#     mean = (torch.as_tensor(kernel_size, dtype=tensor.dtype) - 1) / 2
#     grid = torch.arange(kernel_size, device=tensor.device) - mean
#     kernel_shape = (1, 1, kernel_size)
#     grid = grid.view(kernel_shape)
#     grid = grid.detach()
    
#     source_shape = tensor.shape
#     tensor = torch.movedim(tensor, dim, len(source_shape)-1)
#     dim_last_shape = tensor.shape
    
#     tensor = tensor.reshape(-1, 1, source_shape[dim])
    
#     padding = (math.ceil((kernel_size_int - 1) / 2), math.ceil((kernel_size_int - 1) / 2))
#     tensor_ = F.pad(tensor, padding, padding_mode, padding_value)
    
#     kernel = torch.exp(-0.5 * (grid / sigma) ** 2)
#     kernel = kernel / kernel.sum()
    
#     tensor_ = F.conv1d(tensor_, kernel)
#     tensor_ = tensor_.view(dim_last_shape)
#     tensor_ = torch.movedim(tensor_, len(source_shape)-1, dim)
    
#     return tensor_

def gaussian_filter_1d(tensor, dim, sigma, truncate=4, kernel_size=None, padding_mode='replicate', padding_value=0.0):
    sigma = torch.as_tensor(sigma, device=tensor.device, dtype=tensor.dtype)

    if kernel_size is not None:
        kernel_size = torch.as_tensor(kernel_size, device=tensor.device, dtype=torch.int64)
    else:
        kernel_size = torch.as_tensor(2 * torch.ceil(truncate * sigma) + 1, device=tensor.device, dtype=torch.int64)

    kernel_size = kernel_size.detach()

    kernel_size_int = kernel_size.detach().cpu().numpy()

    mean = (torch.as_tensor(kernel_size, dtype=tensor.dtype) - 1) / 2

    grid = torch.arange(kernel_size, device=tensor.device) - mean

    kernel_shape = (1, 1, kernel_size)
    grid = grid.view(kernel_shape)

    grid = grid.detach()

    source_shape = tensor.shape

    tensor = torch.movedim(tensor, dim, len(source_shape)-1)
    dim_last_shape = tensor.shape
    assert tensor.shape[-1] == source_shape[dim]

    # we need reshape instead of view for batches like B x C x H x W
    tensor = tensor.reshape(-1, 1, source_shape[dim])

    padding = (math.ceil((kernel_size_int - 1) / 2), math.ceil((kernel_size_int - 1) / 2))
    tensor_ = F.pad(tensor, padding, padding_mode, padding_value)

    # create gaussian kernel from grid using current sigma
    kernel = torch.exp(-0.5 * (grid / sigma) ** 2)
    kernel = kernel / kernel.sum()

    # convolve input with gaussian kernel
    tensor_ = F.conv1d(tensor_, kernel)
    tensor_ = tensor_.view(dim_last_shape)
    tensor_ = torch.movedim(tensor_, len(source_shape)-1, dim)

    assert tensor_.shape == source_shape

    return tensor_

class GaussianFilterNd(nn.Module):
    """A differentiable gaussian filter"""

    def __init__(self, dims, sigma, truncate=4, kernel_size=None, padding_mode='replicate', padding_value=0.0,
                 trainable=False):
        """Creates a 1d gaussian filter

        Args:
            dims ([int]): the dimensions to which the gaussian filter is applied. Negative values won't work
            sigma (float): standard deviation of the gaussian filter (blur size)
            input_dims (int, optional): number of input dimensions ignoring batch and channel dimension,
                i.e. use input_dims=2 for images (default: 2).
            truncate (float, optional): truncate the filter at this many standard deviations (default: 4.0).
                This has no effect if the `kernel_size` is explicitely set
            kernel_size (int): size of the gaussian kernel convolved with the input
            padding_mode (string, optional): Padding mode implemented by `torch.nn.functional.pad`.
            padding_value (string, optional): Value used for constant padding.
        """
        # IDEA determine input_dims dynamically for every input
        super(GaussianFilterNd, self).__init__()

        self.dims = dims
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32), requires_grad=trainable)  # default: no optimization
        self.truncate = truncate
        self.kernel_size = kernel_size

        # setup padding
        self.padding_mode = padding_mode
        self.padding_value = padding_value

    def forward(self, tensor):
        """Applies the gaussian filter to the given tensor"""
        for dim in self.dims:
            tensor = gaussian_filter_1d(
                tensor,
                dim=dim,
                sigma=self.sigma,
                truncate=self.truncate,
                kernel_size=self.kernel_size,
                padding_mode=self.padding_mode,
                padding_value=self.padding_value,
            )

        return tensor
    
class Conv2dMultiInput(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        for k, _in_channels in enumerate(in_channels):
            if _in_channels:
                setattr(self, f'conv_part{k}', nn.Conv2d(_in_channels, out_channels, kernel_size, bias=bias))

    def forward(self, tensors):
        assert len(tensors) == len(self.in_channels)

        out = None
        for k, (count, tensor) in enumerate(zip(self.in_channels, tensors)):
            if not count:
                continue
            _out = getattr(self, f'conv_part{k}')(tensor)

            if out is None:
                out = _out
            else:
                out += _out

        return out

#    def extra_repr(self):
#        return f'{self.in_channels}'


class LayerNormMultiInput(nn.Module):
    __constants__ = ['features', 'weight', 'bias', 'eps', 'center', 'scale']

    def __init__(self, features, eps=1e-12, center=True, scale=True):
        super().__init__()
        self.features = features
        self.eps = eps
        self.center = center
        self.scale = scale

        for k, _features in enumerate(features):
            if _features:
                setattr(self, f'layernorm_part{k}', LayerNorm(_features, eps=eps, center=center, scale=scale))

    def forward(self, tensors):
        assert len(tensors) == len(self.features)

        out = []
        for k, (count, tensor) in enumerate(zip(self.features, tensors)):
            if not count:
                assert tensor is None
                out.append(None)
                continue
            out.append(getattr(self, f'layernorm_part{k}')(tensor))

        return out


class Bias(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, tensor):
        return tensor + self.bias[np.newaxis, :, np.newaxis, np.newaxis]

    def extra_repr(self):
        return f'channels={self.channels}'

# class LayerNormMultiInput(nn.Module):
#     __constants__ = ['features', 'weight', 'bias', 'eps', 'center', 'scale']
#     def __init__(self, features, eps=1e-12, center=True, scale=True):
#         super().__init__()
#         self.features = features
#         self.eps = eps
#         self.center = center
#         self.scale = scale

#         for k, _features in enumerate(features):
#             if _features:
#                 setattr(self, f'layernorm_part{k}', LayerNorm(_features, eps=eps, center=center, scale=scale))

#     def forward(self, tensors):
#         assert len(tensors) == len(self.features)
#         out = []
#         for k, (count, tensor) in enumerate(zip(self.features, tensors)):
#             if not count:
#                 assert tensor is None
#                 out.append(None)
#                 continue
#             out.append(getattr(self, f'layernorm_part{k}')(tensor))
#         return out

# class Conv2dMultiInput(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, bias=True):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         for k, _in_channels in enumerate(in_channels):
#             if _in_channels:
#                 setattr(self, f'conv_part{k}', nn.Conv2d(_in_channels, out_channels, kernel_size, bias=bias))

#     def forward(self, tensors):
#         assert len(tensors) == len(self.in_channels)
#         out = None
#         for k, (count, tensor) in enumerate(zip(self.in_channels, tensors)):
#             if not count:
#                 continue
#             _out = getattr(self, f'conv_part{k}')(tensor)
#             if out is None:
#                 out = _out
#             else:
#                 out += _out
#         return out

# class Bias(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.channels = channels
#         self.bias = nn.Parameter(torch.zeros(channels))

#     def forward(self, tensor):
#         return tensor + self.bias[np.newaxis, :, np.newaxis, np.newaxis]

# class GaussianFilterNd(nn.Module):
#     def __init__(self, dims, sigma, truncate=4, kernel_size=None, padding_mode='replicate', padding_value=0.0, trainable=False):
#         super().__init__()
#         self.dims = dims
#         self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32), requires_grad=trainable)
#         self.truncate = truncate
#         self.kernel_size = kernel_size
#         self.padding_mode = padding_mode
#         self.padding_value = padding_value

#     def forward(self, tensor):
#         for dim in self.dims:
#             tensor = gaussian_filter_1d(
#                 tensor,
#                 dim=dim,
#                 sigma=self.sigma,
#                 truncate=self.truncate,
#                 kernel_size=self.kernel_size,
#                 padding_mode=self.padding_mode,
#                 padding_value=self.padding_value,
#             )
#         return tensor