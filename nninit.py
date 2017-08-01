import numpy as np
import torch
from torch.autograd import Variable


def uniform(tensor, a=0, b=1):
    """Fills the input Tensor or Variable with values drawn from a uniform U(a,b)

    Args:
        tensor: a n-dimension torch.Tensor
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.uniform(w)
    """
    if isinstance(tensor, Variable):
        uniform(tensor.data, a=a, b=b)
        return tensor
    else:
        return tensor.uniform_(a, b)


def normal(tensor, mean=0, std=1):
    """Fills the input Tensor or Variable with values drawn from a normal distribution with the given mean and std

    Args:
        tensor: a n-dimension torch.Tensor
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.normal(w)
    """
    if isinstance(tensor, Variable):
        normal(tensor.data, mean=mean, std=std)
        return tensor
    else:
        return tensor.normal_(mean, std)


def constant(tensor, val):
    """Fills the input Tensor or Variable with the value `val`

    Args:
        tensor: a n-dimension torch.Tensor
        val: the value to fill the tensor with

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.constant(w)
    """
    if isinstance(tensor, Variable):
        constant(tensor.data, val)
        return tensor
    else:
        return tensor.fill_(val)


def _calculate_fan_in_and_fan_out(tensor):
    if tensor.ndimension() < 2:
        raise ValueError("fan in and fan out can not be computed for tensor of size ", tensor.size())

    if tensor.ndimension() == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = np.prod(tensor.numpy().shape[2:])
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def xavier_uniform(tensor, gain=1):
    """Fills the input Tensor or Variable with values according to the method described in "Understanding the difficulty of training
       deep feedforward neural networks" - Glorot, X. and Bengio, Y., using a uniform distribution.

       The resulting tensor will have values sampled from U(-a, a) where a = gain * sqrt(2/(fan_in + fan_out))

    Args:
        tensor: a n-dimension torch.Tensor
        gain: an optional scaling factor to be applied

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.xavier_uniform(w, gain=np.sqrt(2.0))
    """
    if isinstance(tensor, Variable):
        xavier_uniform(tensor.data, gain=gain)
        return tensor
    else:
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        std = gain * np.sqrt(2.0 / (fan_in + fan_out))
        a = np.sqrt(3.0) * std
        return tensor.uniform_(-a, a)


def xavier_normal(tensor, gain=1):
    """Fills the input Tensor or Variable with values according to the method described in "Understanding the difficulty of training
       deep feedforward neural networks" - Glorot, X. and Bengio, Y., using a normal distribution.

       The resulting tensor will have values sampled from normal distribution with mean=0 and
       std = gain * sqrt(2/(fan_in + fan_out))

    Args:
        tensor: a n-dimension torch.Tensor
        gain: an optional scaling factor to be applied

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.xavier_normal(w, gain=np.sqrt(2.0))
    """
    if isinstance(tensor, Variable):
        xavier_normal(tensor.data, gain=gain)
        return tensor
    else:
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        std = gain * np.sqrt(2.0 / (fan_in + fan_out))
        return tensor.normal_(0, std)


def kaiming_uniform(tensor, gain=1):
    """Fills the input Tensor or Variable with values according to the method described in "Delving deep into rectifiers: Surpassing
       human-level performance on ImageNet classification" - He, K. et al using a uniform distribution.

       The resulting tensor will have values sampled from U(-a, a) where a = gain * sqrt(1/(fan_in))

    Args:
        tensor: a n-dimension torch.Tensor
        gain: an optional scaling factor to be applied

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.kaiming_uniform(w, gain=np.sqrt(2.0))
    """

    if isinstance(tensor, Variable):
        kaiming_uniform(tensor.data, gain=gain)
        return tensor
    else:
        fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
        std = gain * np.sqrt(1.0 / fan_in)
        a = np.sqrt(3.0) * std
        return tensor.uniform_(-a, a)


def kaiming_normal(tensor, gain=1):
    """Fills the input Tensor or Variable with values according to the method described in "Delving deep into rectifiers:
       Surpassing human-level performance on ImageNet classification" - He, K. et al using a normal distribution.

       The resulting tensor will have values sampled from normal distribution with mean=0 and
       std = gain * sqrt(1/(fan_in))

    Args:
        tensor: a n-dimension torch.Tensor
        gain: an optional scaling factor to be applied

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.kaiming_normal(w, gain=np.sqrt(2.0))
    """
    if isinstance(tensor, Variable):
        kaiming_normal(tensor.data, gain=gain)
        return tensor
    else:
        fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
        std = gain * np.sqrt(1.0 / fan_in)
        return tensor.normal_(0, std)


def orthogonal(tensor, gain=1):
    """Fills the input Tensor or Variable with a (semi) orthogonal matrix. The input tensor must have at least 2 dimensions,
       and for tensors with more than 2 dimensions the trailing dimensions are flattened. viewed as 2D representation with
       rows equal to the first dimension and columns equal to the product of  as a sparse matrix, where the non-zero elements
       will be drawn from a normal distribution with mean=0 and std=`std`.
       Reference: "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks" - Saxe, A. et al.

    Args:
        tensor: a n-dimension torch.Tensor, where n >= 2
        gain: optional gain to be applied

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.orthogonal(w)
    """
    if isinstance(tensor, Variable):
        orthogonal(tensor.data, gain=gain)
        return tensor
    else:
        if tensor.ndimension() < 2:
            raise ValueError("Only tensors with 2 or more dimensions are supported.")

        flattened_shape = (tensor.size(0), int(np.prod(tensor.numpy().shape[1:])))
        flattened = torch.Tensor(flattened_shape[0], flattened_shape[1]).normal_(0, 1)

        u, s, v = np.linalg.svd(flattened.numpy(), full_matrices=False)
        if u.shape == flattened.numpy().shape:
            tensor.view_as(flattened).copy_(torch.from_numpy(u))
        else:
            tensor.view_as(flattened).copy_(torch.from_numpy(v))

        tensor.mul_(gain)
        return tensor


def sparse(tensor, sparsity, std=0.01):
    """Fills the 2D input Tensor or Variable as a sparse matrix, where the non-zero elements will be drawn from a
       normal distribution with mean=0 and std=`std`.

    Args:
        tensor: a n-dimension torch.Tensor
        sparsity: The fraction of elements in each column to be set to zero
        std: the standard deviation of the normal distribution used to generate the non-zero values

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.sparse(w, sparsity=0.1)
    """
    if isinstance(tensor, Variable):
        sparse(tensor.data, sparsity, std=std)
        return tensor
    else:
        if tensor.ndimension() != 2:
            raise ValueError("Sparse initialization only supported for 2D inputs")
        tensor.normal_(0, std)
        rows, cols = tensor.size(0), tensor.size(1)
        num_zeros = int(np.ceil(cols * sparsity))

        for col_idx in range(tensor.size(1)):
            row_indices = np.arange(rows)
            np.random.shuffle(row_indices)
            zero_indices = row_indices[:num_zeros]
            tensor.numpy()[zero_indices, col_idx] = 0

        return tensor