import torch
import torch.nn as nn


class DenseResidualLayer(nn.Module):
    """
    PyTorch like layer for standard linear layer with identity residual connection.
    :param num_features: (int) Number of input / output units for the layer.
    """
    def __init__(self, num_features):
        super(DenseResidualLayer, self).__init__()
        self.linear = nn.Linear(num_features, num_features)

    def forward(self, x):
        """
        Forward-pass through the layer. Implements the following computation:

                f(x) = f_theta(x) + x
                f_theta(x) = W^T x + b

        :param x: (torch.tensor) Input representation to apply layer to ( dim(x) = (batch, num_features) ).
        :return: (torch.tensor) Return f(x) ( dim(f(x) = (batch, num_features) ).
        """
        identity = x
        out = self.linear(x)
        out += identity
        return out


class DenseResidualBlock(nn.Module):
    """
    Wrapping a number of residual layers for residual block. Will be used as building block in FiLM hyper-networks.
    :param in_size: (int) Number of features for input representation.
    :param out_size: (int) Number of features for output representation.
    """
    def __init__(self, in_size, out_size):
        super(DenseResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_size, out_size)
        self.linear2 = nn.Linear(out_size, out_size)
        self.linear3 = nn.Linear(out_size, out_size)
        self.elu = nn.ELU()

    def forward(self, x):
        """
        Forward pass through residual block. Implements following computation:

                h = f3( f2( f1(x) ) ) + x
                or
                h = f3( f2( f1(x) ) )

                where fi(x) = Elu( Wi^T x + bi )

        :param x: (torch.tensor) Input representation to apply layer to ( dim(x) = (batch, in_size) ).
        :return: (torch.tensor) Return f(x) ( dim(f(x) = (batch, out_size) ).
        """
        identity = x
        out = self.linear1(x)
        out = self.elu(out)
        out = self.linear2(out)
        out = self.elu(out)
        out = self.linear3(out)
        if x.shape[-1] == out.shape[-1]:
            out += identity
        return out


class FilmAdaptationNetwork(nn.Module):
    """
    FiLM adaptation network (outputs FiLM adaptation parameters for all layers in a base feature extractor).
    :param layer: (FilmLayerNetwork) Layer object to be used for adaptation.
    :param num_maps_per_layer: (list::int) Number of feature maps for each layer in the network.
    :param num_blocks_per_layer: (list::int) Number of residual blocks in each layer in the network
                                 (see ResNet file for details about ResNet architectures).
    :param z_g_dim: (int) Dimensionality of network input. For this network, z is shared across all layers.
    """
    def __init__(self, layer, num_maps_per_layer, num_blocks_per_layer, z_g_dim):
        super().__init__()
        self.z_g_dim = z_g_dim
        self.num_maps = num_maps_per_layer
        self.num_blocks = num_blocks_per_layer
        self.num_target_layers = len(self.num_maps)
        self.layer = layer
        self.layers = self.get_layers()

    def get_layers(self):
        """
        Loop over layers of base network and initialize adaptation network.
        :return: (nn.ModuleList) ModuleList containing the adaptation network for each layer in base network.
        """
        layers = nn.ModuleList()
        for num_maps, num_blocks in zip(self.num_maps, self.num_blocks):
            layers.append(
                self.layer(
                    num_maps=num_maps,
                    num_blocks=num_blocks,
                    z_g_dim=self.z_g_dim
                )
            )
        return layers

    def forward(self, x):
        """
        Forward pass through adaptation network to create list of adaptation parameters.
        :param x: (torch.tensor) (z -- task level representation for generating adaptation).
        :return: (list::adaptation_params) Returns a list of adaptation dictionaries, one for each layer in base net.
        """
        return [self.layers[layer](x) for layer in range(self.num_target_layers)]

    def regularization_term(self):
        """
        Simple function to aggregate the regularization terms from each of the layers in the adaptation network.
        :return: (torch.scalar) A order-0 torch tensor with the regularization term for the adaptation net params.
        """
        l2_term = 0
        for layer in self.layers:
            l2_term += layer.regularization_term()
        return l2_term


class FilmLayerNetwork(nn.Module):
    """
    Single adaptation network for generating the parameters of each layer in the base network. Will be wrapped around
    by FilmAdaptationNetwork.
    :param num_maps: (int) Number of output maps to be adapted in base network layer.
    :param num_blocks: (int) Number of blocks being adapted in the base network layer.
    :param z_g_dim: (int) Dimensionality of input to network (task level representation).
    """
    def __init__(self, num_maps, num_blocks, z_g_dim):
        super().__init__()
        self.z_g_dim = z_g_dim
        self.num_maps = num_maps
        self.num_blocks = num_blocks

        # Initialize a simple shared layer for all parameter adapters (gammas and betas)
        self.shared_layer = nn.Sequential(
            nn.Linear(self.z_g_dim, self.num_maps),
            nn.ReLU()
        )

        # Initialize the processors (adaptation networks) and regularization lists for each of the output params
        self.gamma1_processors, self.gamma1_regularizers = torch.nn.ModuleList(), torch.nn.ParameterList()
        self.gamma2_processors, self.gamma2_regularizers = torch.nn.ModuleList(), torch.nn.ParameterList()
        self.beta1_processors, self.beta1_regularizers = torch.nn.ModuleList(), torch.nn.ParameterList()
        self.beta2_processors, self.beta2_regularizers = torch.nn.ModuleList(), torch.nn.ParameterList()

        # Generate the required layers / regularization parameters, and collect them in ModuleLists and ParameterLists
        for _ in range(self.num_blocks):
            regularizer = torch.nn.init.normal_(torch.empty(num_maps), 0, 0.001)

            self.gamma1_processors.append(self._make_layer(num_maps))
            self.gamma1_regularizers.append(torch.nn.Parameter(regularizer, requires_grad=True))

            self.beta1_processors.append(self._make_layer(num_maps))
            self.beta1_regularizers.append(torch.nn.Parameter(regularizer, requires_grad=True))

            self.gamma2_processors.append(self._make_layer(num_maps))
            self.gamma2_regularizers.append(torch.nn.Parameter(regularizer, requires_grad=True))

            self.beta2_processors.append(self._make_layer(num_maps))
            self.beta2_regularizers.append(torch.nn.Parameter(regularizer, requires_grad=True))

    @staticmethod
    def _make_layer(size):
        """
        Simple layer generation method for adaptation network of one of the parameter sets (all have same structure).
        :param size: (int) Number of parameters in layer.
        :return: (nn.Sequential) Three layer dense residual network to generate adaptation parameters.
        """
        return nn.Sequential(
            DenseResidualLayer(size),
            nn.ReLU(),
            DenseResidualLayer(size),
            nn.ReLU(),
            DenseResidualLayer(size)
        )

    def forward(self, x):
        """
        Forward pass through adaptation network.
        :param x: (torch.tensor) Input representation to network (task level representation z).
        :return: (list::dictionaries) Dictionary for every block in layer. Dictionary contains all the parameters
                 necessary to adapt layer in base network. Base network is aware of dict structure and can pull params
                 out during forward pass.
        """
        x = self.shared_layer(x)
        block_params = []
        for block in range(self.num_blocks):
            block_param_dict = {
                'gamma1': self.gamma1_processors[block](x).squeeze() * self.gamma1_regularizers[block] +
                          torch.ones_like(self.gamma1_regularizers[block]),
                'beta1': self.beta1_processors[block](x).squeeze() * self.beta1_regularizers[block],
                'gamma2': self.gamma2_processors[block](x).squeeze() * self.gamma2_regularizers[block] +
                          torch.ones_like(self.gamma2_regularizers[block]),
                'beta2': self.beta2_processors[block](x).squeeze() * self.beta2_regularizers[block]
            }
            block_params.append(block_param_dict)
        return block_params

    def regularization_term(self):
        """
        Compute the regularization term for the parameters. Recall, FiLM applies gamma * x + beta. As such, params
        gamma and beta are regularized to unity, i.e. ||gamma - 1||_2 and ||beta||_2.
        :return: (torch.tensor) Scalar for l2 norm for all parameters according to regularization scheme.
        """
        l2_term = 0
        for gamma_regularizer, beta_regularizer in zip(self.gamma1_regularizers, self.beta1_regularizers):
            l2_term += (gamma_regularizer ** 2).sum()
            l2_term += (beta_regularizer ** 2).sum()
        for gamma_regularizer, beta_regularizer in zip(self.gamma2_regularizers, self.beta2_regularizers):
            l2_term += (gamma_regularizer ** 2).sum()
            l2_term += (beta_regularizer ** 2).sum()
        return l2_term


class NullFeatureAdaptationNetwork(nn.Module):
    """
    Dummy adaptation network for the case of "no_adaptation".
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return {}

    @staticmethod
    def regularization_term():
        return 0


class LinearClassifierAdaptationNetwork(nn.Module):
    """
    Versa-style adaptation network for linear classifier (see https://arxiv.org/abs/1805.09921 for full details).
    :param d_theta: (int) Input / output feature dimensionality for layer.
    """
    def __init__(self, d_theta):
        super(LinearClassifierAdaptationNetwork, self).__init__()
        self.weight_means_processor = self._make_mean_dense_block(d_theta, d_theta)
        self.bias_means_processor = self._make_mean_dense_block(d_theta, 1)

    @staticmethod
    def _make_mean_dense_block(in_size, out_size):
        """
        Simple method for generating different types of blocks. Final code only uses dense residual blocks.
        :param in_size: (int) Input representation dimensionality.
        :param out_size: (int) Output representation dimensionality.
        :return: (nn.Module) Adaptation network parameters for outputting classification parameters.
        """
        return DenseResidualBlock(in_size, out_size)

    def forward(self, representation_dict):
        """
        Forward pass through adaptation network. Returns classification parameters for task.
        :param representation_dict: (dict::torch.tensors) Dictionary containing class-level representations for each
                                    class in the task.
        :return: (dict::torch.tensors) Dictionary containing the weights and biases for the classification of each class
                 in the task. Model can extract parameters and build the classifier accordingly. Supports sampling if
                 ML-PIP objective is desired.
        """
        classifier_param_dict = {}
        class_weight_means = []
        class_bias_means = []

        # Extract and sort the label set for the task
        label_set = list(representation_dict.keys())
        label_set.sort()
        num_classes = len(label_set)

        # For each class, extract the representation and pass it through adaptation network to generate classification
        # params for that class. Store parameters in a list,
        for class_num in label_set:
            nu = representation_dict[class_num]
            class_weight_means.append(self.weight_means_processor(nu))
            class_bias_means.append(self.bias_means_processor(nu))

        # Save the parameters as torch tensors (matrix and vector) and add to dictionary
        classifier_param_dict['weight_mean'] = torch.cat(class_weight_means, dim=0)
        classifier_param_dict['bias_mean'] = torch.reshape(torch.cat(class_bias_means, dim=1), [num_classes, ])

        return classifier_param_dict


class FilmArAdaptationNetwork(nn.Module):
    """
    Auto-Regressive FiLM adaptation network (outputs FiLM adaptation parameters for all layers in a base
    feature extractor). Similar to FilmAdaptation network, but forward pass leverages Auto-regressive information.
    :param feature_extractor: (nn.Module) Base network for adaptation (used in AR pass).
    :param num_maps_per_layer: (list::int) Number of feature maps for each layer in the network.
    :param num_blocks_per_layer: (list::int) Number of residual blocks in each layer in the network
                                 (see ResNet file for details about ResNet architecures).
    :param num_initial_conv_maps: (int) Number of maps from initial conv layer in base network.
    :param z_g_dim: (int) Dimensionality of network input. For this network, z is shared across all layers.
    """
    def __init__(self, feature_extractor, num_maps_per_layer, num_blocks_per_layer, num_initial_conv_maps, z_g_dim):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.z_g_dim = z_g_dim
        self.num_maps = num_maps_per_layer
        self.num_blocks = num_blocks_per_layer
        self.num_target_layers = len(self.num_maps)
        self.affine_layer = FilmArLayerNetwork
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.layers = nn.ModuleList()
        previous_maps = [num_initial_conv_maps] + self.num_maps
        for input_dim, num_maps, num_blocks in zip(previous_maps[:-1], self.num_maps, self.num_blocks):
            self.layers.append(
                self.affine_layer(
                    input_dim=input_dim,
                    z_g_dim=self.z_g_dim,
                    num_maps=num_maps,
                    num_blocks=num_blocks
                )
            )

    def forward(self, x, task_representation):
        """
        Forward pass through the adaptation network. Implements auto-regressive computation detailed in paper (see
        Section 2.2 in https://arxiv.org/pdf/1906.07697 for further details).
        :param x: (torch.tensor) Example images context set of task.
        :param task_representation: (torch.tensor) Global task representation z_G from set encoder.
        :return: (list::dict::torch.tensor) List of dictionaries of adaptation parameters to be used by model.
        """
        def flatten(t):
            t = self.avgpool(t)
            return t.view(t.size(0), -1)

        param_dicts = []
        # Start with initial convolution layer from ResNet to embedd context set and generate first local representation
        z = self.feature_extractor.get_layer_output(x, None, 0)
        z_hn = flatten(z)
        # For every following layer: pass global and local representations through hypernet layer. This returns the
        # next layer adaptation parameters. Use these to make a pass through the next layer with context set, and
        # save adaptation parameters.
        for layer, hn_layer in enumerate(self.layers):
            param_dicts.append(hn_layer(z_hn, task_representation))
            z = self.feature_extractor.get_layer_output(z, param_dicts, layer + 1)
            z_hn = flatten(z)
        return param_dicts

    def regularization_term(self):
        """
        Simple function to aggregate the regularization terms from each of the layers in the adaptation network.
        :return: (torch.scalar) A order-0 torch tensor with the regularization term for the adaptation net params.
        """
        l2_term = 0
        for layer in self.layers:
            l2_term += layer.regularization_term()
        return l2_term


class FilmArLayerNetwork(nn.Module):
    """
    Single adaptation network for generating the parameters of each layer in the base network. Will be wrapped around
    by FilmARAdaptationNetwork. Generates adaptation parameters for next layer give z_G and z_AR.
    :param input_dim: (int) Dimensionality of local representation.
    :param num_maps: (int) Number of output maps to be adapted in base network layer.
    :param num_blocks: (int) Number of blocks being adapted in the base network layer.
    :param z_g_dim: (int) Dimensionality of input to network (task level representation).
    """
    def __init__(self, input_dim, z_g_dim, num_maps, num_blocks):
        super().__init__()
        self.input_dim = input_dim
        self.z_g_dim = z_g_dim
        self.num_maps = num_maps
        self.num_blocks = num_blocks
        self.shared_layer, self.shared_layer_post = self.get_shared_layers()

        # Initialize ModuleLists and ParameterLists for layer processeors (hyper-nets) and regluarizers
        self.gamma1_processors, self.gamma1_regularizers = torch.nn.ModuleList(), torch.nn.ParameterList()
        self.gamma2_processors, self.gamma2_regularizers = torch.nn.ModuleList(), torch.nn.ParameterList()
        self.beta1_processors, self.beta1_regularizers = torch.nn.ModuleList(), torch.nn.ParameterList()
        self.beta2_processors, self.beta2_regularizers = torch.nn.ModuleList(), torch.nn.ParameterList()

        # Loop over blocks. For each block, collect necessary parameters and regularizers
        for _ in range(self.num_blocks):
            regularizer = torch.nn.init.normal_(torch.empty(num_maps), 0, 0.001)

            self.gamma1_processors.append(self._make_layer(self.num_maps + self.z_g_dim, num_maps))
            self.gamma1_regularizers.append(torch.nn.Parameter(regularizer, requires_grad=True))

            self.beta1_processors.append(self._make_layer(self.num_maps + self.z_g_dim, num_maps))
            self.beta1_regularizers.append(torch.nn.Parameter(regularizer, requires_grad=True))

            self.gamma2_processors.append(self._make_layer(self.num_maps + self.z_g_dim, num_maps))
            self.gamma2_regularizers.append(torch.nn.Parameter(regularizer, requires_grad=True))

            self.beta2_processors.append(self._make_layer(self.num_maps + self.z_g_dim, num_maps))
            self.beta2_regularizers.append(torch.nn.Parameter(regularizer, requires_grad=True))

    def get_shared_layers(self):
        """
        Simple layer generation method for shared layer to be used in layer adaptation network.
        :param size: (int) Number of parameters in layer.
        :return: (nn.Sequential) Three layer dense residual network to generate adaptation parameters.
        """
        shared_layer_pre = nn.Sequential(
            nn.Linear(self.input_dim, self.num_maps),
            nn.ReLU(),
            DenseResidualLayer(self.num_maps),
            nn.ReLU(),
            DenseResidualLayer(self.num_maps),
            nn.ReLU(),
            DenseResidualLayer(self.num_maps)
        )
        shared_layer_post = nn.Sequential(
            nn.Linear(self.num_maps, self.num_maps),
            nn.ReLU()
        )
        return shared_layer_pre, shared_layer_post

    @staticmethod
    def _make_layer(in_size, out_size):
        """
        Simple layer generation method for processor for each of the parameters associated with the base net layer.
        :param size: (int) Number of parameters in layer.
        :return: (nn.Sequential) Three layer dense residual network to generate adaptation parameters.
        """
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
            DenseResidualLayer(out_size),
            nn.ReLU(),
            DenseResidualLayer(out_size),
            nn.ReLU(),
            DenseResidualLayer(out_size)
        )

    def forward(self, x, task_representation):
        """
        Forward pass through adaptation network.
        :param x: (torch.tensor) Input representation to network (task level representation z).
        :return: (list::dictionaries) Dictionary for every block in layer. Dictionary contains all the parameters
                 necessary to adapt layer in base network. Base network is aware of dict structure and can pull params
                 out during forward pass.
        """
        x = self.shared_layer(x)
        x = torch.mean(x, dim=0, keepdim=True)
        x = self.shared_layer_post(x)
        x = torch.cat([x, task_representation], dim=-1)
        block_params = []
        for block in range(self.num_blocks):
            block_param_dict = {
                'gamma1': self.gamma1_processors[block](x).squeeze() * self.gamma1_regularizers[block] +
                          torch.ones_like(self.gamma1_regularizers[block]),
                'beta1': self.beta1_processors[block](x).squeeze() * self.beta1_regularizers[block],
                'gamma2': self.gamma2_processors[block](x).squeeze() * self.gamma2_regularizers[block] +
                          torch.ones_like(self.gamma2_regularizers[block]),
                'beta2': self.beta2_processors[block](x).squeeze() * self.beta2_regularizers[block]
            }
            block_params.append(block_param_dict)
        return block_params

    def regularization_term(self):
        """
        Compute the regularization term for the parameters. Recall, FiLM applies gamma * x + beta. As such, params
        gamma and beta are regularized to unity, i.e. ||gamma - 1||_2 and ||beta||_2.
        :return: (torch.tensor) Scalar for l2 norm for all parameters according to regularization scheme.
        """
        l2_term = 0
        for gamma_regularizer, beta_regularizer in zip(self.gamma1_regularizers, self.beta1_regularizers):
            l2_term += (gamma_regularizer ** 2).sum()
            l2_term += (beta_regularizer ** 2).sum()
        for gamma_regularizer, beta_regularizer in zip(self.gamma2_regularizers, self.beta2_regularizers):
            l2_term += (gamma_regularizer ** 2).sum()
            l2_term += (beta_regularizer ** 2).sum()
        return l2_term
