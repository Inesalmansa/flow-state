import torch
from torch.nn import functional as F

import numpy as np

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def unconstrained_rational_quadratic_spline(inputs,unnormalized_widths,unnormalized_heights,unnormalized_derivatives,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,):

    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask
    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)
    if tails == "linear":
        unnormalized_derivatives_ = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives_[..., 0] = constant
        unnormalized_derivatives_[..., -1] = constant
        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    elif tails[0] == "circular":
        unnormalized_derivatives_ = F.pad(unnormalized_derivatives, pad=(0, 1))
        unnormalized_derivatives_[..., -1] = unnormalized_derivatives_[..., 0]
        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    elif isinstance(tails, list) or isinstance(tails, tuple):
        unnormalized_derivatives_ = unnormalized_derivatives.clone()
        ind_lin = [t == "linear" for t in tails]
        ind_circ = [t == "circular" for t in tails]
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives_[..., ind_lin, 0] = constant
        unnormalized_derivatives_[..., ind_lin, -1] = constant
        unnormalized_derivatives_[..., ind_circ, -1] = unnormalized_derivatives_[..., ind_circ, 0]
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    if torch.is_tensor(tail_bound):
        tail_bound_ = torch.broadcast_to(tail_bound, inputs.shape)
        left = -tail_bound_[inside_interval_mask]
        right = tail_bound_[inside_interval_mask]
        bottom = -tail_bound_[inside_interval_mask]
        top = tail_bound_[inside_interval_mask]
    else:
        left = -tail_bound
        right = tail_bound
        bottom = -tail_bound
        top = tail_bound

    (
        outputs_masked,
        logabsdet_masked
    ) = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        # unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives_[inside_interval_mask, :],
        inverse=inverse,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )
    if outputs.dtype == outputs_masked.dtype and logabsdet.dtype == logabsdet_masked.dtype:
        outputs[inside_interval_mask] = outputs_masked
        logabsdet[inside_interval_mask] = logabsdet_masked
    else:
        outputs[inside_interval_mask] = outputs_masked.to(outputs.dtype)
        logabsdet[inside_interval_mask] = logabsdet_masked.to(logabsdet.dtype)

    return outputs, logabsdet


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    num_bins = unnormalized_widths.shape[-1]

    if torch.is_tensor(left):
        lim_tensor = True
    else:
        lim_tensor = False

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    if lim_tensor:
        cumwidths = (right[..., None] - left[..., None]) * cumwidths + left[..., None]
    else:
        cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    if lim_tensor:
        cumheights = (top[..., None] - bottom[..., None]) * cumheights + bottom[
            ..., None
        ]
    else:
        cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = abs(b.pow(2) - 4 * a * c)
        if (discriminant < 0).any():
            problematic_indices = (discriminant < 0).nonzero(as_tuple=True)
            problematic_values = discriminant[problematic_indices]
            raise ValueError(f"Discriminant is less than zero at indices {problematic_indices} with values {problematic_values}, which indicates complex roots.")
        if torch.isnan(discriminant).any():
            print("NaN detected in discriminant computation")
            # Optionally, print or log values that contribute to it
            print("a:", a)
            print("b:", b)
            print("c:", c)
            print("discriminant:", discriminant)
            raise ValueError("Discriminant computation resulted in NaN.")
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, logabsdet
