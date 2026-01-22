import torch
import torch.nn as nn
import re
import math
import warnings
from collections import OrderedDict
from einops import rearrange
from networks.simplenerf import PositionalEncoding


# https://github.com/jihoontack/GradNCP/blob/main/models/metamodule/metamodule.py
class MetaModule(nn.Module):
    """
    Base class for PyTorch meta-learning modules. These modules accept an
    additional argument `params` in their `forward` method.

    Notes
    -----
    Objects inherited from `MetaModule` are fully compatible with PyTorch
    modules from `torch.nn.Module`. The argument `params` is a dictionary of
    tensors, with full support of the computation graph (for differentiation).
    """

    def __init__(self):
        super(MetaModule, self).__init__()
        self._children_modules_parameters_cache = dict()

    def meta_named_parameters(self, prefix="", recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items() if isinstance(module, MetaModule) else [],
            prefix=prefix,
            recurse=recurse,
        )
        for elem in gen:
            yield elem

    def meta_parameters(self, recurse=True):
        for name, param in self.meta_named_parameters(recurse=recurse):
            yield param

    def get_subdict(self, params, key=None):
        if params is None:
            return None

        all_names = tuple(params.keys())
        if (key, all_names) not in self._children_modules_parameters_cache:
            if key is None:
                self._children_modules_parameters_cache[(key, all_names)] = all_names

            else:
                key_escape = re.escape(key)
                key_re = re.compile(r"^{0}\.(.+)".format(key_escape))

                self._children_modules_parameters_cache[(key, all_names)] = [
                    key_re.sub(r"\1", k) for k in all_names if key_re.match(k) is not None
                ]

        names = self._children_modules_parameters_cache[(key, all_names)]
        if not names:
            warnings.warn(
                "Module `{0}` has no parameter corresponding to the "
                "submodule named `{1}` in the dictionary `params` "
                "provided as an argument to `forward()`. Using the "
                "default parameters for this submodule. The list of "
                "the parameters in `params`: [{2}].".format(
                    self.__class__.__name__, key, ", ".join(all_names)
                ),
                stacklevel=2,
            )
            return None

        return OrderedDict([(name, params[f"{key}.{name}"]) for name in names])


class MetaSequential(nn.Sequential, MetaModule):
    __doc__ = nn.Sequential.__doc__

    def forward(self, input, params=None):
        for name, module in self._modules.items():
            if isinstance(module, MetaModule):
                input = module(input, params=self.get_subdict(params, name))
            elif isinstance(module, nn.Module):
                input = module(input)
            else:
                raise TypeError(
                    "The module must be either a torch module "
                    "(inheriting from `nn.Module`), or a `MetaModule`. "
                    "Got type: `{0}`".format(type(module))
                )
        return input


class MetaBatchLinear(nn.Linear, MetaModule):
    """
    A linear meta-layer that can deal with batched weight matrices and biases,
    as for instance output by a hypernetwork.
    inputs:  [batch_size, num_grids, dim_in]
    params: [batch_size, dim_out, dim_in]
    """

    __doc__ = nn.Linear.__doc__

    def forward(self, inputs, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
            for name, param in params.items():
                params[name] = param[None, ...].repeat((inputs.size(0),) + (1,) * len(param.shape))

        bias = params.get("bias", None)
        weight = params["weight"]

        inputs = rearrange(inputs, "b i o -> b o i")
        output = torch.bmm(weight, inputs)
        output = rearrange(output, "b i o -> b o i")

        if bias is not None:
            output += bias.unsqueeze(-2)
        return output


class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class MetaSirenLayer(MetaModule):
    """
    Single layer of SIREN; uses SIREN-style init. scheme.
    """

    def __init__(self, dim_in, dim_out, w0=30.0, c=6.0, is_first=False, is_final=False):
        super().__init__()
        # Encapsulates MetaLinear and activation.
        self.linear = MetaBatchLinear(dim_in, dim_out)
        self.activation = nn.Identity() if is_final else Sine(w0)
        # Initializes according to SIREN init.
        self.init_(c=c, w0=w0, is_first=is_first)

    def init_(self, c, w0, is_first):
        dim_in = self.linear.weight.size(1)
        w_std = 1 / dim_in if is_first else (math.sqrt(c / dim_in) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        nn.init.uniform_(self.linear.bias, -w_std, w_std)

    def forward(self, x, params=None):
        return self.activation(self.linear(x, self.get_subdict(params, "linear")))


class MetaSiren(MetaModule):
    """
    SIREN as a meta-network.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers=4,
        w0=30.0,
        w0_initial=30.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.w0 = w0
        layers = []
        for ind in range(num_layers - 1):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden
            layers.append(
                MetaSirenLayer(
                    dim_in=layer_dim_in, dim_out=dim_hidden, w0=layer_w0, is_first=is_first
                )
            )
        layers.append(MetaSirenLayer(dim_in=dim_hidden, dim_out=dim_out, w0=w0, is_final=True))
        self.layers = MetaSequential(*layers)

    def forward(self, x, params=None):
        return self.layers(x, params=self.get_subdict(params, "layers")) + 0.5


class MetaSirenPenultimate(MetaModule):
    """
    SIREN as a meta-network.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers=4,
        w0=30.0,
        w0_initial=30.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.w0 = w0
        layers = []
        for ind in range(num_layers - 1):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden
            layers.append(
                MetaSirenLayer(
                    dim_in=layer_dim_in, dim_out=dim_hidden, w0=layer_w0, is_first=is_first
                )
            )
        self.layers = MetaSequential(*layers)
        self.last_layer = MetaSirenLayer(dim_in=dim_hidden, dim_out=dim_out, w0=w0, is_final=True)

    def forward(self, x, params=None, get_features=False):
        feature = self.layers(x, params=self.get_subdict(params, "layers"))
        out = self.last_layer(feature, params=self.get_subdict(params, "last_layer")) + 0.5

        if get_features:
            return out, feature
        else:
            return out


class MetaReLULayer(MetaModule):
    # def __init__(self, dim_in, dim_out, w0=30.0, c=6.0, is_first=False, is_final=False):
    def __init__(self, dim_in, dim_out, is_final=False):
        super().__init__()
        # Encapsulates MetaLinear and activation.
        self.linear = MetaBatchLinear(dim_in, dim_out)
        self.activation = nn.Identity() if is_final else nn.ReLU()

    # ? init using defualt reset_parameters method in nn.Linear (TODO: check is it ok)
    #     self.init_(c=c, w0=w0, is_first=is_first)

    # def init_(self, c, w0, is_first):
    #     dim_in = self.linear.weight.size(1)
    #     w_std = 1 / dim_in if is_first else (math.sqrt(c / dim_in) / w0)
    #     nn.init.uniform_(self.linear.weight, -w_std, w_std)
    #     nn.init.uniform_(self.linear.bias, -w_std, w_std)

    def forward(self, x, params=None):
        return self.activation(self.linear(x, self.get_subdict(params, "linear")))


class MetaReLU(MetaModule):
    """
    A simple NeRF MLP without view dependence and skip connections.
    """

    def __init__(
        self, in_features, max_freq, num_freqs, hidden_features, hidden_layers, out_features
    ):
        assert hidden_layers >= 2
        super().__init__()

        self.net = []
        self.net.append(PositionalEncoding(max_freq, num_freqs))
        self.net.append(MetaReLULayer(2 * num_freqs * in_features, hidden_features))

        for i in range(hidden_layers - 2):
            self.net.append(MetaReLULayer(hidden_features, hidden_features))

        self.net.append(MetaReLULayer(hidden_features, out_features, is_final=True))
        self.net = MetaSequential(*self.net)

    def forward(self, x, params=None):
        """
        At each input xyz point return the rgb and sigma values.
        Input:
            x: (num_rays, num_samples, 3)
        Output:
            rgb: (num_rays, num_samples, 3)
            sigma: (num_rays, num_samples)
        """
        # TODO: check if +0.5 is needed (not in the simplenerf code)
        out = self.net(x, params=self.get_subdict(params, "net")) + 0.5
        rgb = torch.sigmoid(out[..., :-1])
        sigma = torch.nn.functional.softplus(out[..., -1])
        return rgb, sigma
