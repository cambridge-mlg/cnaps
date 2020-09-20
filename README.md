# CNAPs: Fast and Flexible Multi-Task Classification Using Conditional Neural Adaptive Processes
This repository contains the code to reproduce the few-shot classification experiments carried out in
[Fast and Flexible Multi-Task Classification Using Conditional Neural Adaptive Processes](https://arxiv.org/abs/1906.07697)
and [TASKNORM: Rethinking Batch Normalization for Meta-Learning](https://arxiv.org/pdf/2003.03284.pdf).

The code has been authored by: John Bronskill, Jonathan Gordon, and James Reqeima.

## Dependencies
This code requires the following:
* Python 3.5 or greater
* PyTorch 1.0 or greater
* TensorFlow 1.14 or greater

## GPU Requirements
* To train or test a CNAPs model with auto-regressive FiLM adaptation on Meta-Dataset, 2 GPUs with 16GB or more memory
are required.
* To train or test a CNAPs model with FiLM only adaptation plus TaskNorm on Meta-Dataset, 2 GPUs with 16GB or more memory
are required.
* It is not currently possible to run a CNAPs model with auto-regressive FiLM adaptation plus TaskNorm on Meta-Dataset
(even using 2 GPUs with 16GB of memory). It may be possible (we have not tried) to run this configuration on 2 GPUs with
24GB of memory.
* The other modes require only a single GPU with at least 16 GB of memory.
* If you want to run any of the modes on a single GPU, you can train on a single dataset with fixed shot and way.
If shot and way are not too large, this configuration will require a single GPU with less than 16GB of memory.
An example command line is (though this will not reproduce the meta-dataset results):

```python run_cnaps.py --feature_adaptation film -i 20000 -lr 0.001 --batch_normalization task_norm-i -- dataset omniglot --way 5 --shot 5 --data_path <path to directory containing Meta-Dataset records>```

## Installation
1. Clone or download this repository.
2. Configure Meta-Dataset:
    * Follow the the "User instructions" in the Meta-Dataset repository (https://github.com/google-research/meta-dataset)
    for "Installation" and "Downloading and converting datasets". This will take some time.
3. Install additional test datasets (MNIST, CIFAR10, CIFAR100):
    * Change to the $DATASRC directory: ```cd $DATASRC```
    * Download the MNIST test images: ```wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz```
    * Download the MNIST test labels: ```wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz```
    * Download the CIFAR10 dataset: ```wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz```
    * Extract the CIFAR10 dataset: ```tar -zxvf cifar-10-python.tar.gz```
    * Download the CIFAR100 dataset: ```wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz```
    * Extract the CIFAR10 dataset: ```tar -zxvf cifar-100-python.tar.gz```
    * Change to the ```cnaps/src``` directory in the repository.
    * Run: ```python prepare_extra_datasets.py```

## Usage
To train and test CNAPs on Meta-Dataset:

1. First run the following two commands.
    
    ```ulimit -n 50000```

    ```export META_DATASET_ROOT=<root directory of the cloned or downloaded Meta-Dataset repository>```
    
    Note the above commands need to be run every time you open a new command shell.
    
2. Execute the ```run_cnaps.py``` script from the ```src``` directory following the instructions at the beginning
of the file.

## Expected Results
Below are the results extracted from our papers. The results will vary from run to run due by a percent or two up or 
down due to the fact that the Meta-Dataset reader generates different tasks each run and the CNAPs networks have random
sets of initial parameters. The FiLM + TaskNorm configuration consistently yields the best results and trains in much
less time than the other configurations.

**Models trained on all datasets**

| Dataset       | No Adaptation | FiLM         | FiLM + AR     | FiLM + TaskNorm |
| ---           | ---           | ---          | ---           | ---             |
| ILSVRC        | 43.8±1.0      | **51.3±1.0** | **52.3±1.0**  | **50.6±1.1** |
| Omniglot      | 60.1±1.3      | 88.0±0.7     | 88.4±0.7      | **90.7±0.6** |
| Aircraft      | 53.0±0.9      | 76.8±0.8     | 80.5±0.6      | **83.8±0.6** |
| Birds         | 55.7±1.0      | 71.4±0.9     | 72.2±0.9      | **74.6±0.8** |
| Textures      | 60.5±0.8      | **62.5±0.7** | 58.3±0.7      | **62.1±0.7** |
| Quick Draw    | 58.1±1.0      | 71.9±0.8     | 72.5±0.8      | **74.8±0.7** |
| Fungi         | 28.6±0.9      | 46.0±1.1     | 47.4±1.0      | **48.7±1.0** |
| VGG Flower    | 75.3±0.7      | **89.2±0.5** | 86.0±0.5      | **89.6±0.5** |
| Traffic Signs | 55.0±0.9      | 60.1±0.9     | 60.2±0.9      | **67.0±0.7** |
| MSCOCO        | 41.2±1.0      | **42.0±1.0** | **42.6±1.1**  | **43.4±1.0** |
| MNIST         | 76.0±0.8      | 88.6±0.5     | **92.7±0.4**  | **92.3±0.4** |
| CIFAR10       | 61.5±0.7      | 60.0±0.8     | 61.5±0.7      | **69.3±0.8** |
| CIFAR100      | 44.8±1.0      | 48.1±1.0     | 50.1±1.0      | **54.6±1.1** |

## Contact
To ask questions or report issues, please open an issue on the issues tracker.

## Citation
If you use this code, please cite our [CNAPs](https://arxiv.org/abs/1906.07697) and [TaskNorm](https://arxiv.org/pdf/2003.03284.pdf) papers:
```
@incollection{requeima2019cnaps,
  title      = {Fast and Flexible Multi-Task Classification using Conditional Neural Adaptive Processes},
  author     = {Requeima, James and Gordon, Jonathan and Bronskill, John and Nowozin, Sebastian and Turner, Richard E},
  booktitle  = {Advances in Neural Information Processing Systems 32},
  editor     = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\' Alch\'{e}-Buc and E. Fox and R. Garnett},
  pages      = {7957--7968},
  year       = {2019},
  publisher  = {Curran Associates, Inc.},
}

@incollection{bronskill2020tasknorm,
  title     = {TaskNorm: Rethinking Batch Normalization for Meta-Learning},
  author    = {Bronskill, John and Gordon, Jonathan and Requeima, James and Nowozin, Sebastian and Turner, Richard},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning},
  volume    = {119},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  year      = {2020}
}
```
