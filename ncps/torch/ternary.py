# Copyright 2022 Mathias Lechner and Ramin Hasani
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_


class _TernaryQuantize(torch.autograd.Function):
    """Straight-Through Estimator (STE) for ternary weight quantization."""

    @staticmethod
    def forward(ctx, weight, threshold_factor=0.7):
        threshold = threshold_factor * weight.abs().mean()
        ternary = torch.sign(weight) * (weight.abs() >= threshold).float()
        ctx.save_for_backward(weight)
        return ternary

    @staticmethod
    def backward(ctx, grad_output):
        (weight,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Soft-clip gradients where |w| > 1.0
        grad_input = torch.where(
            weight.abs() > 1.0, grad_input * 0.5, grad_input
        )
        return grad_input, None


def ternary_quantize(weight, threshold_factor=0.7):
    """Quantize weights to ternary values {-1, 0, +1} using STE.

    :param weight: Weight tensor to quantize
    :param threshold_factor: Fraction of mean absolute weight used as threshold (default 0.7)
    :return: Ternary weight tensor
    """
    return _TernaryQuantize.apply(weight, threshold_factor)


class TernaryLinear(nn.Module):
    """Linear layer with ternary weight quantization {-1, 0, +1}.

    Uses a Straight-Through Estimator (STE) for training. Each output channel
    has a learnable per-channel scale factor.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        threshold_factor=0.7,
        quantize=True,
    ):
        super(TernaryLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_factor = threshold_factor
        self.quantize = quantize

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        xavier_uniform_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Learnable per-output-channel scale factor
        self.scale = nn.Parameter(torch.ones(out_features))

    def forward(self, x):
        if self.quantize:
            w_ternary = ternary_quantize(self.weight, self.threshold_factor)
            w_scaled = w_ternary * self.scale.unsqueeze(1)
            return F.linear(x, w_scaled, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)

    def get_ternary_weights(self):
        """Returns ternary weight tensor without gradient tracking."""
        with torch.no_grad():
            return ternary_quantize(self.weight, self.threshold_factor)

    def get_weight_distribution(self):
        """Returns dict with counts of -1, 0, +1 values and sparsity."""
        w = self.get_ternary_weights()
        total = w.numel()
        neg_one = (w == -1).sum().item()
        zero = (w == 0).sum().item()
        pos_one = (w == 1).sum().item()
        return {
            "neg_one": neg_one,
            "zero": zero,
            "pos_one": pos_one,
            "total": total,
            "sparsity": zero / total if total > 0 else 0.0,
        }
