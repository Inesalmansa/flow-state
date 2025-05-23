import torch
from torch import nn
import numpy as np

from ..base import Flow
from .coupling import PiecewiseRationalQuadraticCoupling
from .autoregressive import MaskedPiecewiseRationalQuadraticAutoregressive
from ...nets.resnet import ResidualNet
from ...nets.Transformer import TransformerNet
from ...nets.graph_network import FullEquivariantGraphNetwork
from ...utils.masks import create_alternating_binary_mask
from ...utils.nn import PeriodicFeaturesElementwise
from ...utils.splines import DEFAULT_MIN_DERIVATIVE


class CoupledRationalQuadraticSpline(Flow):
    """
    Neural spline flow coupling layer, wrapper for the implementation
    of Durkan et al., see [source](https://github.com/bayesiains/nsf)
    """

    def __init__(
        self,
        num_input_channels,
        num_blocks,
        num_hidden_channels,
        num_context_channels=None,
        num_bins=8,
        tails="linear",
        tail_bound=3.0,
        activation=nn.ReLU,
        dropout_probability=0.0,
        reverse_mask=False,
        init_identity=True,
    ):
        """Constructor

        Args:
          num_input_channels (int): Flow dimension
          num_blocks (int): Number of residual blocks of the parameter NN
          num_hidden_channels (int): Number of hidden units of the NN
          num_context_channels (int): Number of context/conditional channels
          num_bins (int): Number of bins
          tails (str): Behaviour of the tails of the distribution, can be linear, circular for periodic distribution, or None for distribution on the compact interval
          tail_bound (float): Bound of the spline tails
          activation (torch module): Activation function
          dropout_probability (float): Dropout probability of the NN
          reverse_mask (bool): Flag whether the reverse mask should be used
          init_identity (bool): Flag, initialize transform as identity
        """
        super().__init__()

        def transform_net_create_fn(in_features, out_features):
            net = ResidualNet(
                in_features=in_features,
                out_features=out_features,
                context_features=num_context_channels,
                hidden_features=num_hidden_channels,
                num_blocks=num_blocks,
                activation=activation(),
                dropout_probability=dropout_probability,
                use_batch_norm=False,
            )
            # net = FullEquivariantGraphNetwork(
            #     num_node=in_features,
            #     out_dim=out_features,
            #     feat_dim=2,
            #     hidden_dim=num_hidden_channels,
            #     num_layers=num_blocks,
            #     dropout=dropout_probability
            # )
            if init_identity:
                torch.nn.init.constant_(net.final_layer.weight, 0.0)
                torch.nn.init.constant_(
                    net.final_layer.bias, np.log(np.exp(1 - DEFAULT_MIN_DERIVATIVE) - 1)
                )
            return net

        self.prqct = PiecewiseRationalQuadraticCoupling(
            mask=create_alternating_binary_mask(num_input_channels, even=reverse_mask),
            transform_net_create_fn=transform_net_create_fn,
            num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,
            # Setting True corresponds to equations (4), (5), (6) in the NSF paper:
            apply_unconditional_transform=True,
        )

    def forward(self, z, context=None):
        z, log_det = self.prqct.inverse(z, context)
        return z, log_det.view(-1)

    def inverse(self, z, context=None):
        z, log_det = self.prqct(z, context)
        return z, log_det.view(-1)


class CircularCoupledRationalQuadraticSpline(Flow):
    """
    Neural spline flow coupling layer with circular coordinates
    """

    def __init__(
        self,
        num_input_channels,
        num_blocks,
        num_hidden_channels,
        ind_circ,
        num_heads=4,
        num_context_channels=None,
        num_bins=8,
        tail_bound=3.0,
        net_type = "residual",
        activation=nn.ReLU,
        dropout_probability=0.0,
        reverse_mask=False,
        mask=None,
        init_identity=True,
    ):
        """Constructor

        Args:
          num_input_channels (int): Flow dimension
          num_blocks (int): Number of residual blocks of the parameter NN
          num_hidden_channels (int): Number of hidden units of the NN
          num_context_channels (int): Number of context/conditional channels
          ind_circ (Iterable): Indices of the circular coordinates
          num_bins (int): Number of bins
          tail_bound (float or Iterable): Bound of the spline tails
          net_type (str): which net implementation (residual, gnn, transformer), default is residual
          activation (torch module): Activation function
          dropout_probability (float): Dropout probability of the NN
          reverse_mask (bool): Flag whether the reverse mask should be used
          mask (torch tensor): Mask to be used, alternating masked generated is None
          init_identity (bool): Flag, initialize transform as identity
        """
        super().__init__()
        

        if mask is None:
            mask = create_alternating_binary_mask(num_input_channels, even=reverse_mask)
        #这里是创建了间隔的掩码
        features_vector = torch.arange(num_input_channels)
        identity_features = features_vector.masked_select(mask <= 0)
        ind_circ = torch.tensor(ind_circ)
        ind_circ_id = []
        for i, id in enumerate(identity_features):
            if id in ind_circ:
                ind_circ_id += [i]

        if torch.is_tensor(tail_bound):
            scale_pf = np.pi / tail_bound[ind_circ_id]
        else:
            scale_pf = np.pi / tail_bound

        # defining three different transform_net_create_fn functions

        def residual_transform_net_create_fn(in_features, out_features):
            if len(ind_circ_id) > 0:
                    # pf = None
                    # ndim (int): number of dimensions
                    # ind (iterable): indices of input elements to convert to periodic features
                    # scale: Scalar or iterable, used to scale inputs before converting them to periodic features
                    pf = PeriodicFeaturesElementwise(in_features, ind_circ_id, scale_pf)
            else:
                pf = None

            net = ResidualNet(
                    in_features=2*in_features,
                    # in_features=in_features,
                    out_features=out_features,
                    context_features=num_context_channels,
                    hidden_features=num_hidden_channels,
                    num_blocks=num_blocks,
                    activation=activation(),
                    dropout_probability=dropout_probability,
                    use_batch_norm=True,
                    preprocessing=pf,
                )

            if init_identity:
                torch.nn.init.constant_(net.final_layer.weight, 0.0)
                torch.nn.init.constant_(
                    net.final_layer.bias, np.log(np.exp(1 - DEFAULT_MIN_DERIVATIVE) - 1)
                )
            print(f"Using network: {net}")
            return net

        def gnn_transform_net_create_fn(in_features, out_features):
            if len(ind_circ_id) > 0:
                    # pf = None
                    # ndim (int): number of dimensions
                    # ind (iterable): indices of input elements to convert to periodic features
                    # scale: Scalar or iterable, used to scale inputs before converting them to periodic features
                    pf = PeriodicFeaturesElementwise(in_features, ind_circ_id, scale_pf)
            else:
                pf = None

            net = FullEquivariantGraphNetwork(
                        num_node = in_features,
                        out_dim=out_features,
                        feat_dim = 2,
                        hidden_dim = num_hidden_channels,
                        num_layers = num_blocks,
                        dropout = dropout_probability,
                        preprocessing = pf
                    )

            # if init_identity:
            #     torch.nn.init.constant_(net.final_layer.weight, 0.0)
            #     torch.nn.init.constant_(
            #         net.final_layer.bias, np.log(np.exp(1 - DEFAULT_MIN_DERIVATIVE) - 1)
            #     )
            print(f"Using network: {net}")
            return net

        def transformer_transform_net_create_fn(in_features, out_features):
            if len(ind_circ_id) > 0:
                    # pf = None
                    # ndim (int): number of dimensions
                    # ind (iterable): indices of input elements to convert to periodic features
                    # scale: Scalar or iterable, used to scale inputs before converting them to periodic features
                    pf = PeriodicFeaturesElementwise(in_features, ind_circ_id, scale_pf)
            else:
                pf = None

            net = TransformerNet(
                    in_features = 2*in_features,
                    out_features = out_features,
                    embed_dim = num_hidden_channels,
                    num_heads = num_heads,
                    num_layers = num_blocks,
                    preprocessing = pf
                )
            # if init_identity:
            #     torch.nn.init.constant_(net.final_layer.weight, 0.0)
            #     torch.nn.init.constant_(
            #         net.final_layer.bias, np.log(np.exp(1 - DEFAULT_MIN_DERIVATIVE) - 1)
            #     )
            print(f"Using network: {net}")
            return net

        if net_type == "residual":
            passed_transform_net_create_fn = residual_transform_net_create_fn
        
        elif net_type == "gnn":
            passed_transform_net_create_fn = gnn_transform_net_create_fn
        
        elif net_type == "transformer":
            passed_transform_net_create_fn = transformer_transform_net_create_fn
        
        else:
            passed_transform_net_create_fn = residual_transform_net_create_fn

           
        tails = [
            "circular" if i in ind_circ else "linear" for i in range(num_input_channels)
        ]

        self.prqct = PiecewiseRationalQuadraticCoupling(
            mask=mask,
            transform_net_create_fn=passed_transform_net_create_fn,
            num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,
            apply_unconditional_transform=True,
        )

    def forward(self, z, context=None):
        z, log_det = self.prqct.inverse(z, context)
        return z, log_det.view(-1)

    def inverse(self, z, context=None):
        z, log_det = self.prqct(z, context)
        return z, log_det.view(-1)


class AutoregressiveRationalQuadraticSpline(Flow):
    """
    Neural spline flow coupling layer, wrapper for the implementation
    of Durkan et al., see [sources](https://github.com/bayesiains/nsf)
    """

    def __init__(
        self,
        num_input_channels,
        num_blocks,
        num_hidden_channels,
        num_context_channels=None,
        num_bins=8,
        tail_bound=3,
        activation=nn.ReLU,
        dropout_probability=0.0,
        permute_mask=False,
        init_identity=True,
    ):
        """Constructor

        Args:
          num_input_channels (int): Flow dimension
          num_blocks (int): Number of residual blocks of the parameter NN
          num_hidden_channels (int): Number of hidden units of the NN
          num_context_channels (int): Number of context/conditional channels
          num_bins (int): Number of bins
          tail_bound (int): Bound of the spline tails
          activation (torch.nn.Module): Activation function
          dropout_probability (float): Dropout probability of the NN
          permute_mask (bool): Flag, permutes the mask of the NN
          init_identity (bool): Flag, initialize transform as identity
        """
        super().__init__()

        self.mprqat = MaskedPiecewiseRationalQuadraticAutoregressive(
            features=num_input_channels,
            hidden_features=num_hidden_channels,
            context_features=num_context_channels,
            num_bins=num_bins,
            tails="linear",
            tail_bound=tail_bound,
            num_blocks=num_blocks,
            use_residual_blocks=True,
            random_mask=False,
            permute_mask=permute_mask,
            activation=activation(),
            dropout_probability=dropout_probability,
            use_batch_norm=False,
            init_identity=init_identity,
        )

    def forward(self, z, context=None):
        z, log_det = self.mprqat.inverse(z, context=context)
        return z, log_det.view(-1)

    def inverse(self, z, context=None):
        z, log_det = self.mprqat(z, context=context)
        return z, log_det.view(-1)


class CircularAutoregressiveRationalQuadraticSpline(Flow):
    """
    Neural spline flow coupling layer, wrapper for the implementation
    of Durkan et al., see [sources](https://github.com/bayesiains/nsf)
    """

    def __init__(
        self,
        num_input_channels,
        num_blocks,
        num_hidden_channels,
        ind_circ,
        num_context_channels=None,
        num_bins=8,
        tail_bound=3,
        activation=nn.ReLU,
        dropout_probability=0.0,
        permute_mask=True,
        init_identity=True,
    ):
        """Constructor

        Args:
          num_input_channels (int): Flow dimension
          num_blocks (int): Number of residual blocks of the parameter NN
          num_hidden_channels (int): Number of hidden units of the NN
          ind_circ (Iterable): Indices of the circular coordinates
          num_context_channels (int): Number of context/conditional channels
          num_bins (int): Number of bins
          tail_bound (int): Bound of the spline tails
          activation (torch module): Activation function
          dropout_probability (float): Dropout probability of the NN
          permute_mask (bool): Flag, permutes the mask of the NN
          init_identity (bool): Flag, initialize transform as identity
        """
        super().__init__()

        tails = [
            "circular" if i in ind_circ else "linear" for i in range(num_input_channels)
        ]

        self.mprqat = MaskedPiecewiseRationalQuadraticAutoregressive(
            features=num_input_channels,
            hidden_features=num_hidden_channels,
            context_features=num_context_channels,
            num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,
            num_blocks=num_blocks,
            use_residual_blocks=True,
            random_mask=False,
            permute_mask=permute_mask,
            activation=activation(),
            dropout_probability=dropout_probability,
            use_batch_norm=False,
            init_identity=init_identity,
        )

    def forward(self, z, context=None):
        z, log_det = self.mprqat.inverse(z, context=context)
        return z, log_det.view(-1)

    def inverse(self, z, context=None):
        z, log_det = self.mprqat(z, context=context)
        return z, log_det.view(-1)
