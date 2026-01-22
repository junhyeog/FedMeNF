# https://github.com/sanowar-raihan/nerf-meta/blob/main/models/nerf.py
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat


class PositionalEncoding(nn.Module):
    """
    Positional Encoding of the input coordinates.

    encodes x to (..., sin(2^k x), cos(2^k x), ...)
    k takes "num_freqs" number of values equally spaced between [0, max_freq]
    """

    def __init__(self, max_freq, num_freqs):
        """
        Args:
            max_freq (int): maximum frequency in the positional encoding.
            num_freqs (int): number of frequencies between [0, max_freq]
        """
        super().__init__()
        freqs = 2 ** torch.linspace(0, max_freq, num_freqs)
        self.register_buffer("freqs", freqs)  # (num_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (num_rays, num_samples, in_features)
        Outputs:
            out: (num_rays, num_samples, 2*num_freqs*in_features)
        """
        x_proj = x.unsqueeze(dim=-2) * self.freqs.unsqueeze(dim=-1)  # (num_rays, num_samples, num_freqs, in_features)
        x_proj = x_proj.reshape(*x.shape[:-1], -1)  # (num_rays, num_samples, num_freqs*in_features)
        out = torch.cat(
            [torch.sin(x_proj), torch.cos(x_proj)], dim=-1
        )  # (num_rays, num_samples, 2*num_freqs*in_features)
        return out


class SimpleNeRF(nn.Module):
    """
    A simple NeRF MLP without view dependence and skip connections.
    """

    def __init__(self, in_features, max_freq, num_freqs, hidden_features, hidden_layers, out_features):
        """
        in_features: number of features in the input.
        max_freq: maximum frequency in the positional encoding.
        num_freqs: number of frequencies between [0, max_freq] in the positional encoding.
        hidden_features: number of features in the hidden layers of the MLP.
        hidden_layers: number of hidden layers.
        out_features: number of features in the output.

        #
        https://github.com/tancik/learnit/blob/main/Experiments/shapenet.ipynb
        x = np.concatenate([np.concatenate([np.sin(coords*(2**i)), np.cos(coords*(2**i))], axis=-1) for i in np.linspace(0,8,20)]
        max_freq = 8
        num_freqs = 20
        """
        super().__init__()

        self.net = []
        self.net.append(PositionalEncoding(max_freq, num_freqs))
        self.net.append(nn.Linear(2 * num_freqs * in_features, hidden_features))
        self.net.append(nn.ReLU())

        for i in range(hidden_layers - 2):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.ReLU())

        self.net.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        """
        At each input xyz point return the rgb and sigma values.
        Input:
            x: (num_rays, num_samples, 3)
        Output:
            rgb: (num_rays, num_samples, 3)
            sigma: (num_rays, num_samples)
        """
        out = self.net(x)
        rgb = torch.sigmoid(out[..., :-1])
        sigma = F.softplus(out[..., -1])
        return rgb, sigma


# def build_nerf(args):
#     model = SimpleNeRF(
#         in_features=3,
#         max_freq=args.max_freq,
#         num_freqs=args.num_freqs,
#         hidden_features=args.hidden_features,
#         hidden_layers=args.hidden_layers,
#         out_features=4,
#     )
#     return model


class IPCNeRF(nn.Module):
    """
    Generalizable Implicit Neural Representations with Instance Pattern Composers
    Compatible with functional programming using functional_call and vmap.
    """

    def __init__(
        self,
        in_features,
        max_freq,
        num_freqs,
        hidden_features,
        hidden_layers,
        out_features,
        rank,
        modulated_layer_idxs=None,
    ):
        """
        Args:
            in_features (int): Number of input features.
            max_freq (int): Maximum frequency for positional encoding.
            num_freqs (int): Number of frequency bands.
            hidden_features (int): Number of features in the hidden layers.
            hidden_layers (int): Total number of layers.
            out_features (int): Number of output features.
            rank (int): Rank for low-rank modulation.
            modulated_layer_idxs (list): Indices of layers to be modulated.
        """
        super().__init__()

        if modulated_layer_idxs is None:
            modulated_layer_idxs = []

        self.modulated_layer_idxs = modulated_layer_idxs
        self.rank = rank

        self.pos_encoding = PositionalEncoding(max_freq, num_freqs)

        self.in_dim = 2 * num_freqs * in_features  # After positional encoding

        # Build MLP layers
        self.layers = nn.ModuleList()
        self.layer_names = []
        for idx in range(hidden_layers):
            if idx == 0:
                input_dim = self.in_dim
            else:
                input_dim = hidden_features
            if idx == hidden_layers - 1:
                output_dim = out_features
            else:
                output_dim = hidden_features

            layer = nn.Linear(input_dim, output_dim)
            self.layers.append(layer)
            self.layer_names.append(f"layer{idx}")

        self.activation = nn.ReLU()

        # Build U, V matrices for modulated layers
        self.u_params, self.modulation_factors = self.init_u_v(
            self.layers, self.layer_names, self.modulated_layer_idxs, self.rank
        )

    def init_u_v(self, layers, layer_names, modulated_layer_idxs, rank):
        params = OrderedDict()
        modulation_factors = OrderedDict()
        # Initialize U matrices for modulated layers
        for idx in modulated_layer_idxs:
            layer_name = layer_names[idx]
            input_dim = layers[idx].in_features
            output_dim = layers[idx].out_features
            U = torch.randn(rank, output_dim) / (rank * input_dim) ** 0.5
            U.requires_grad_()
            params[f"{layer_name}_U"] = U

            V = torch.randn(input_dim, rank)
            V.requires_grad_()
            modulation_factors[f"{layer_name}_V"] = V

        return params, modulation_factors

    def forward(self, x, params, modulation_factors=None):
        """
        Forward pass with low-rank modulation.

        Args:
            x (Tensor): Input coordinates, shape (batch_size, num_rays, num_samples, in_features).
            params (OrderedDict): Model parameters (including U matrices).
            modulation_factors (OrderedDict): Modulation factors V for specified layers.
                Keys should be 'layer{idx}_V'. Each V is of shape [batch_size, input_dim, rank].

        Returns:
            rgb (Tensor): RGB values.
            sigma (Tensor): Density values.
        """

        x = self.pos_encoding(x)  # Apply positional encoding
        batch_size, num_rays, num_samples, input_dim = x.shape
        x = x.view(batch_size, -1, input_dim)  # Squeeze rays and samples

        # Forward pass through layers
        for idx, layer in enumerate(self.layers):
            layer_name = self.layer_names[idx]
            weight_key = f"{layer_name}.weight"
            bias_key = f"{layer_name}.bias"

            if weight_key in params:
                weight = params[weight_key]
                bias = params[bias_key]
            else:
                weight = layer.weight
                bias = layer.bias

            # Apply modulation if specified
            if modulation_factors is not None and idx in self.modulated_layer_idxs:
                # U is in params
                U_key = f"{layer_name}_U"
                U = params[U_key]  # [batch_size, rank, fan_out]
                V = modulation_factors[f"{layer_name}_V"]  # [batch_size, fan_in, rank]

                W = torch.bmm(V, U)
                W = F.normalize(W, dim=1)  # Optinal: Normalize Weight

                # Linear transformation
                x = torch.bmm(x, W) + bias.unsqueeze(0).unsqueeze(1)
            else:
                weight = weight.mean(0)
                bias = bias.mean(0)
                # Standard linear layer
                x = F.linear(x, weight, bias)

            if idx < len(self.layers) - 1:
                x = self.activation(x)

        x = x.view(batch_size, num_rays, num_samples, -1)  # Extend to Original Dim
        out = x
        rgb = torch.sigmoid(out[..., :-1])  # RGB
        sigma = F.softplus(out[..., -1])  # Density (sigma)
        return rgb, sigma
