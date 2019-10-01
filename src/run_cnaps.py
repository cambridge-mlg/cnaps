"""
Script to reproduce the few-shot classification results on Meta-Dataset in:
"Fast and Flexible Multi-Task Classification Using Conditional Neural Adaptive Processes"
https://arxiv.org/pdf/1906.07697.pdf

The following command lines should reproduce the published results within error-bars:

Note before running any of the commands, you need to run the following two commands:

ulimit -n 50000

export META_DATASET_ROOT=<root directory of the cloned or downloaded Meta-Dataset repository>

CNAPs using auto-regressive FiLM adaptation, meta-training on all datasets
--------------------------------------------------------------------------
python run_cnaps.py --data_path <path to directory containing Meta-Dataset records>

CNAPs using FiLM adaptation only, meta-training on all datasets
---------------------------------------------------------------
python run_cnaps.py --feature_adaptation film --data_path <path to directory containing Meta-Dataset records>

CNAPs using no feature adaptation, meta-training on all datasets
----------------------------------------------------------------
python run_cnaps.py --feature_adaptation no_adaptation --data_path <path to directory containing Meta-Dataset records>

Note that when using auto-regressive FiLM adaptation, 2 GPUs with at least 16GB of memory are required.
The other modes require only a single GPU with at least 16 GB of memory.

"""


import torch
import numpy as np
import argparse
import os
import sys
from utils import print_and_log, get_log_files, ValidationAccuracies, loss, aggregate_accuracy
from model import Cnaps
from meta_dataset_reader import MetaDatasetReader
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet TensorFlow warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Quiet TensorFlow warnings

NUM_TRAIN_TASKS = 110000
NUM_VALIDATION_TASKS = 200
NUM_TEST_TASKS = 600
VALIDATION_FREQUENCY = 10000


def main():
    learner = Learner()
    learner.run()


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()

        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_log_files(self.args.checkpoint_dir)

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        gpu_device = 'cuda:0'
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        self.train_set, self.validation_set, self.test_set = self.init_data()
        self.metadataset = MetaDatasetReader(self.args.data_path, self.args.mode, self.train_set, self.validation_set,
                                             self.test_set)
        self.loss = loss
        self.accuracy_fn = aggregate_accuracy
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.optimizer.zero_grad()
        self.validation_accuracies = ValidationAccuracies(self.validation_set)

    def init_model(self):
        use_two_gpus = self.use_two_gpus()
        model = Cnaps(device=self.device, use_two_gpus=use_two_gpus, args=self.args).to(self.device)
        model.train()  # set encoder is always in train mode to process context data
        model.feature_extractor.eval()  # feature extractor is always in eval mode
        if use_two_gpus:
            model.distribute_model()
        return model

    def init_data(self):

        train_set = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower']
        validation_set = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower',
                          'mscoco']
        test_set = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower',
                    'traffic_sign', 'mscoco', 'mnist', 'cifar10', 'cifar100']

        return train_set, validation_set, test_set

    """
    Command line parser
    """
    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--data_path", default="../datasets", help="Path to dataset records.")
        parser.add_argument("--pretrained_resnet_path", default="../models/pretrained_resnet.pt.tar",
                            help="Path to pretrained feature extractor model.")
        parser.add_argument("--mode", choices=["train", "test", "train_test"], default="train_test",
                            help="Whether to run training only, testing only, or both training and testing.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=5e-4, help="Learning rate.")
        parser.add_argument("--tasks_per_batch", type=int, default=16,
                            help="Number of tasks between parameter optimizations.")
        parser.add_argument("--checkpoint_dir", "-c", default='../checkpoints', help="Directory to save checkpoint to.")
        parser.add_argument("--test_model_path", "-m", default=None, help="Path to model to load and test.")
        parser.add_argument("--feature_adaptation", choices=["no_adaptation", "film", "film+ar"], default="film+ar",
                            help="Method to adapt feature extractor parameters.")

        args = parser.parse_args()

        return args

    def run(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as session:
            if self.args.mode == 'train' or self.args.mode == 'train_test':
                train_accuracies = []
                losses = []
                total_iterations = NUM_TRAIN_TASKS
                for iteration in range(total_iterations):
                    torch.set_grad_enabled(True)
                    task_dict = self.metadataset.get_train_task(session)
                    task_loss, task_accuracy = self.train_task(task_dict)
                    train_accuracies.append(task_accuracy)
                    losses.append(task_loss)

                    # optimize
                    if ((iteration + 1) % self.args.tasks_per_batch == 0) or (iteration == (total_iterations - 1)):
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    if (iteration + 1) % 1000 == 0:
                        # print training stats
                        print_and_log(self.logfile,'Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}'
                                      .format(iteration + 1, total_iterations, torch.Tensor(losses).mean().item(),
                                              torch.Tensor(train_accuracies).mean().item()))
                        train_accuracies = []
                        losses = []

                    if ((iteration + 1) % VALIDATION_FREQUENCY == 0) and (iteration + 1) != total_iterations:
                        # validate
                        accuracy_dict = self.validate(session)
                        self.validation_accuracies.print(self.logfile, accuracy_dict)
                        # save the model if validation is the best so far
                        if self.validation_accuracies.is_better(accuracy_dict):
                            self.validation_accuracies.replace(accuracy_dict)
                            torch.save(self.model.state_dict(), self.checkpoint_path_validation)
                            print_and_log(self.logfile, 'Best validation model was updated.')
                            print_and_log(self.logfile, '')

                # save the final model
                torch.save(self.model.state_dict(), self.checkpoint_path_final)

            if self.args.mode == 'train_test':
                self.test(self.checkpoint_path_final, session)
                self.test(self.checkpoint_path_validation, session)

            if self.args.mode == 'test':
                self.test(self.args.test_model_path, session)

            self.logfile.close()

    def train_task(self, task_dict):
        context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict)

        target_logits = self.model(context_images, context_labels, target_images)
        task_loss = self.loss(target_logits, target_labels, self.device) / self.args.tasks_per_batch
        if self.args.feature_adaptation == 'film' or self.args.feature_adaptation == 'film+ar':
            if self.use_two_gpus():
                regularization_term = (self.model.feature_adaptation_network.regularization_term()).cuda(0)
            else:
                regularization_term = (self.model.feature_adaptation_network.regularization_term())
            regularizer_scaling = 0.001
            task_loss += regularizer_scaling * regularization_term
        task_accuracy = self.accuracy_fn(target_logits, target_labels)

        task_loss.backward(retain_graph=False)

        return task_loss, task_accuracy

    def validate(self, session):
        with torch.no_grad():
            accuracy_dict ={}
            for item in self.validation_set:
                accuracies = []
                for _ in range(NUM_VALIDATION_TASKS):
                    task_dict = self.metadataset.get_validation_task(item, session)
                    context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict)
                    target_logits = self.model(context_images, context_labels, target_images)
                    accuracy = self.accuracy_fn(target_logits, target_labels)
                    accuracies.append(accuracy.item())
                    del target_logits

                accuracy = np.array(accuracies).mean() * 100.0
                confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

                accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence}

        return accuracy_dict

    def test(self, path, session):
        self.model = self.init_model()
        self.model.load_state_dict(torch.load(path))

        print_and_log(self.logfile, "")  # add a blank line
        print_and_log(self.logfile, 'Testing model {0:}: '.format(path))

        with torch.no_grad():
            for item in self.test_set:
                accuracies = []
                for _ in range(NUM_TEST_TASKS):
                    task_dict = self.metadataset.get_test_task(item, session)
                    context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict)
                    target_logits = self.model(context_images, context_labels, target_images)
                    accuracy = self.accuracy_fn(target_logits, target_labels)
                    accuracies.append(accuracy.item())
                    del target_logits

                accuracy = np.array(accuracies).mean() * 100.0
                accuracy_confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

                print_and_log(self.logfile, '{0:}: {1:3.1f}+/-{2:2.1f}'.format(item, accuracy, accuracy_confidence))

    def prepare_task(self, task_dict):
        context_images_np, context_labels_np = task_dict['context_images'], task_dict['context_labels']
        target_images_np, target_labels_np = task_dict['target_images'], task_dict['target_labels']

        context_images_np = context_images_np.transpose([0, 3, 1, 2])
        context_images_np, context_labels_np = self.shuffle(context_images_np, context_labels_np)
        context_images = torch.from_numpy(context_images_np)
        context_labels = torch.from_numpy(context_labels_np)

        target_images_np = target_images_np.transpose([0, 3, 1, 2])
        target_images_np, target_labels_np = self.shuffle(target_images_np, target_labels_np)
        target_images = torch.from_numpy(target_images_np)
        target_labels = torch.from_numpy(target_labels_np)

        context_images = context_images.to(self.device)
        target_images = target_images.to(self.device)
        context_labels = context_labels.to(self.device)
        target_labels = target_labels.type(torch.LongTensor).to(self.device)

        return context_images, target_images, context_labels, target_labels

    def shuffle(self, images, labels):
        """
        Return shuffled data.
        """
        permutation = np.random.permutation(images.shape[0])
        return images[permutation], labels[permutation]

    def use_two_gpus(self):
        use_two_gpus = False
        if self.args.feature_adaptation == "film+ar":
            use_two_gpus = True  # film+ar model does not fit on one GPU, so use model parallelism
        return use_two_gpus


if __name__ == "__main__":
    main()
