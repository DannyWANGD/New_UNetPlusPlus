import torch
from torch import nn

from nnunet.network_architecture.generic_UNetPlusPlus import Generic_UNetPlusPlus
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.network_training.nnUNetPlusPlusTrainerV2 import nnUNetPlusPlusTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper


class nnUNetPlusPlusTrainerV2_MLKA(nnUNetPlusPlusTrainerV2):
    """UNet++ V2 trainer with MLKA refinement enabled for 2D experiments."""

    def initialize_network(self):
        if self.threeD:
            raise NotImplementedError("nnUNetPlusPlusTrainerV2_MLKA currently supports 2D training only.")

        conv_op = nn.Conv2d
        dropout_op = nn.Dropout2d
        norm_op = nn.InstanceNorm2d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

        self.network = Generic_UNetPlusPlus(self.num_input_channels, self.base_num_features, self.num_classes,
                                            len(self.net_num_pool_op_kernel_sizes),
                                            self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs,
                                            net_nonlin, net_nonlin_kwargs, True, False, lambda x: x,
                                            InitWeights_He(1e-2),
                                            self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False,
                                            True, True,
                                            use_mlka=True, mlka_groups=3, mlka_norm='instance')
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
