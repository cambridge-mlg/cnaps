from resnet import film_resnet18, resnet18
from adaptation_networks import NullFeatureAdaptationNetwork, FilmAdaptationNetwork, \
    LinearClassifierAdaptationNetwork, FilmLayerNetwork, FilmArAdaptationNetwork
from set_encoder import SetEncoder
from utils import linear_classifier


""" Creates the set encoder, feature extractor, feature adaptation, classifier, and classifier adaptation networks.
"""


class ConfigureNetworks:
    def __init__(self, pretrained_resnet_path, feature_adaptation):

        self.classifier = linear_classifier

        self.encoder = SetEncoder()
        z_g_dim = self.encoder.pre_pooling_fn.output_size

        # parameters for ResNet18
        num_maps_per_layer = [64, 128, 256, 512]
        num_blocks_per_layer = [2, 2, 2, 2]
        num_initial_conv_maps = 64

        if feature_adaptation == "no_adaptation":
            self.feature_extractor = resnet18(
                pretrained=True,
                pretrained_model_path=pretrained_resnet_path
            )
            self.feature_adaptation_network = NullFeatureAdaptationNetwork()

        elif feature_adaptation == "film":
            self.feature_extractor = film_resnet18(
                pretrained=True,
                pretrained_model_path=pretrained_resnet_path
            )
            self.feature_adaptation_network = FilmAdaptationNetwork(
                layer=FilmLayerNetwork,
                num_maps_per_layer=num_maps_per_layer,
                num_blocks_per_layer=num_blocks_per_layer,
                z_g_dim=z_g_dim
            )

        elif feature_adaptation == 'film+ar':
            self.feature_extractor = film_resnet18(
                pretrained=True,
                pretrained_model_path=pretrained_resnet_path
            )
            self.feature_adaptation_network = FilmArAdaptationNetwork(
                feature_extractor=self.feature_extractor,
                num_maps_per_layer=num_maps_per_layer,
                num_blocks_per_layer=num_blocks_per_layer,
                num_initial_conv_maps = num_initial_conv_maps,
                z_g_dim=z_g_dim
            )

        # Freeze the parameters of the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.classifier_adaptation_network = LinearClassifierAdaptationNetwork(self.feature_extractor.output_size)

    def get_encoder(self):
        return self.encoder

    def get_classifier(self):
        return self.classifier

    def get_classifier_adaptation(self):
        return self.classifier_adaptation_network

    def get_feature_adaptation(self):
        return self.feature_adaptation_network

    def get_feature_extractor(self):
        return self.feature_extractor
