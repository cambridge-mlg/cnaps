import torch
import torch.nn as nn
from collections import OrderedDict
from utils import split_first_dim_linear
from config_networks import ConfigureNetworks
from set_encoder import mean_pooling

NUM_SAMPLES=1


class Cnaps(nn.Module):
    """
    Main model class. Implements several CNAPs models (with / without feature adaptation, with /without auto-regressive
    adaptation parameters generation.
    :param device: (str) Device (gpu or cpu) on which model resides.
    :param use_two_gpus: (bool) Whether to paralleize the model (model parallelism) across two GPUs.
    :param args: (Argparser) Arparse object containing model hyper-parameters.
    """
    def __init__(self, device, use_two_gpus, args):
        super(Cnaps, self).__init__()
        self.args = args
        self.device = device
        self.use_two_gpus = use_two_gpus
        networks = ConfigureNetworks(pretrained_resnet_path=self.args.pretrained_resnet_path,
                                     feature_adaptation=self.args.feature_adaptation,
                                     batch_normalization=args.batch_normalization)
        self.set_encoder = networks.get_encoder()
        self.classifier_adaptation_network = networks.get_classifier_adaptation()
        self.classifier = networks.get_classifier()
        self.feature_extractor = networks.get_feature_extractor()
        self.feature_adaptation_network = networks.get_feature_adaptation()
        self.task_representation = None
        self.class_representations = OrderedDict()  # Dictionary mapping class label (integer) to encoded representation
        self.total_iterations = args.training_iterations

    def forward(self, context_images, context_labels, target_images):
        """
        Forward pass through the model for one episode.
        :param context_images: (torch.tensor) Images in the context set (batch x C x H x W).
        :param context_labels: (torch.tensor) Labels for the context set (batch x 1 -- integer representation).
        :param target_images: (torch.tensor) Images in the target set (batch x C x H x W).
        :return: (torch.tensor) Categorical distribution on label set for each image in target set (batch x num_labels).
        """
        # extract train and test features
        self.task_representation = self.set_encoder(context_images)
        context_features, target_features = self._get_features(context_images, target_images)

        # get the parameters for the linear classifier.
        self._build_class_reps(context_features, context_labels)
        classifier_params = self._get_classifier_params()

        # classify
        sample_logits = self.classifier(target_features, classifier_params)
        self.class_representations.clear()

        # this adds back extra first dimension for num_samples
        return split_first_dim_linear(sample_logits, [NUM_SAMPLES, target_images.shape[0]])

    def _get_features(self, context_images, target_images):
        """
        Helper function to extract task-dependent feature representation for each image in both context and target sets.
        :param context_images: (torch.tensor) Images in the context set (batch x C x H x W).
        :param target_images: (torch.tensor) Images in the target set (batch x C x H x W).
        :return: (tuple::torch.tensor) Feature representation for each set of images.
        """
        # Parallelize forward pass across multiple GPUs (model parallelism)
        if self.use_two_gpus:
            context_images_1 = context_images.cuda(1)
            target_images_1 = target_images.cuda(1)
            if self.args.feature_adaptation == 'film+ar':
                task_representation_1 = self.task_representation.cuda(1)
                # Get adaptation params by passing context set through the adaptation networks
                self.set_batch_norm_mode(True)
                self.feature_extractor_params = self.feature_adaptation_network(context_images_1, task_representation_1)
            else:
                task_representation_1 = self.task_representation.cuda(1)
                # Get adaptation params by passing context set through the adaptation networks
                self.feature_extractor_params = self.feature_adaptation_network(task_representation_1)
            # Given adaptation parameters for task, conditional forward pass through the adapted feature extractor
            self.set_batch_norm_mode(True)
            context_features_1 = self.feature_extractor(context_images_1, self.feature_extractor_params)
            context_features = context_features_1.cuda(0)
            self.set_batch_norm_mode(False)
            target_features_1 = self.feature_extractor(target_images_1, self.feature_extractor_params)
            target_features = target_features_1.cuda(0)
        else:
            if self.args.feature_adaptation == 'film+ar':
                # Get adaptation params by passing context set through the adaptation networks
                self.set_batch_norm_mode(True)
                self.feature_extractor_params = self.feature_adaptation_network(context_images, self.task_representation)
            else:
                # Get adaptation params by passing context set through the adaptation networks
                self.feature_extractor_params = self.feature_adaptation_network(self.task_representation)
            # Given adaptation parameters for task, conditional forward pass through the adapted feature extractor
            self.set_batch_norm_mode(True)
            context_features = self.feature_extractor(context_images, self.feature_extractor_params)
            self.set_batch_norm_mode(False)
            target_features = self.feature_extractor(target_images, self.feature_extractor_params)

        return context_features, target_features

    def _build_class_reps(self, context_features, context_labels):
        """
        Construct and return class level representation for each class in task.
        :param context_features: (torch.tensor) Adapted feature representation for each image in the context set.
        :param context_labels: (torch.tensor) Label for each image in the context set.
        :return: (void) Updates the internal class representation dictionary.
        """
        for c in torch.unique(context_labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(context_features, 0, self._extract_class_indices(context_labels, c))
            class_rep = mean_pooling(class_features)
            self.class_representations[c.item()] = class_rep

    def _get_classifier_params(self):
        """
        Processes the class representations and generated the linear classifier weights and biases.
        :return: Linear classifier weights and biases.
        """
        classifier_params = self.classifier_adaptation_network(self.class_representations)
        return classifier_params

    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

    def distribute_model(self):
        """
        Moves the feature extractor and feature adaptation network to a second GPU.
        :return: Nothing
        """
        self.feature_extractor.cuda(1)
        self.feature_adaptation_network.cuda(1)

    def set_batch_norm_mode(self, context):
        """
        Controls the batch norm mode in the feature extractor.
        :param context: Set to true ehen processing the context set and False when processing the target set.
        :return: Nothing
        """
        if self.args.batch_normalization == "basic":
            self.feature_extractor.eval()  # always in eval mode
        else:
            # "task_norm-i" - respect context flag, regardless of state
            if context:
                self.feature_extractor.train()  # use train when processing the context set
            else:
                self.feature_extractor.eval()  # use eval when processing the target set
