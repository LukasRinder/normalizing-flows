from data.dataset_loader import *
from enum import Enum
from normalizingflows.flow_manager import FlowType
from data.toy_data import generate_2d_data
from data.dataset_loader import load_and_preprocess_uci
from data.dataset_loader import load_and_preprocess_mnist
from data.dataset_loader import load_and_preprocess_celeb
from utils.train_utils import shuffle_split
from utils.types import DataType

class Dataset():
    def __init__(self, dataset_name, batch_size, data_size=2000, category=-1):
        self.name = dataset_name.lower()
        self.batch_size = batch_size

        if dataset_name in ["swissroll", "circles", "rings", "moons", "4gaussians", "8gaussians", "pinwheel", "2spirals", "checkerboard", "line", "cos", "tum", "random_toy_data"]:
                    # get train data and perform a train-validation-test split
            train_split = 0.8
            val_split = 0.1
            samples, interval = generate_2d_data(dataset_name, batch_size=data_size)
            train_data, self.batched_val_data, self.batched_test_data = shuffle_split(samples, train_split, val_split)
            train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
            self.batched_train_data = train_dataset.batch(batch_size)
            self.intervals = [interval, interval]
            self.data_type = DataType.toydata
            self.input_output_shape = (2, 2)
            
        elif dataset_name in ["power", "gas", "miniboone", "hepmass"]:
            self.batched_train_data, self.batched_val_data, self.batched_test_data, self.intervals = load_and_preprocess_uci(batch_size=batch_size, shuffle=True)
            self.data_type = DataType.uci
            sample_batch = next(iter(self.batched_train_data))
            input_shape = sample_batch.shape[1]
            self.input_output_shape = (input_shape, input_shape)
            uci_trainsizes = {"power": 1659917,
                  "gas": 852174,
                  "hepmass": 315123,
                  "miniboone": 29556,
                  "bsds300": 1000000}
            self.dataset_size = uci_trainsizes[dataset_name]
        
        elif dataset_name == "mnist":
            self.batched_train_data, self.batched_val_data, self.batched_test_data, interval = load_and_preprocess_mnist(logit_space=True, batch_size=128, shuffle=True, classes=category, channels=False)
            self.data_type = DataType.mnist
            sample_batch = next(iter(self.batched_train_data))
            # assumes channels first
            if sample_batch.shape[-1] == sample_batch.shape[-2]:
                size = sample_batch.shape[-1]
                input_shape = size * size
            self.intervals = [interval for _ in range(input_shape)]
            self.input_output_shape = (input_shape, (size, size))
            self.dataset_size = 50000

        #celeb should be proccessed while training
        elif dataset_name == "celeb":
            self.data_type = DataType.celeb
            self.batched_train_data, self.batched_val_data, self.batched_test_data, interval = load_celeb(logit_space=True, batch_size=128, shuffle=True)
            
            # assumes batch size first
            sample_batch = next(iter(batched_train_data))
            celeb_shape = sample_batch["image"].shape[1:]
            input_shape = celeb_shape[0] * celeb_shape[1] * celeb_shape[2]
            
            self.intervals = [interval for _ in range(input_shape)]
            self.input_output_shape = (input_shape, (size, size))
            self.dataset_size = 202599
                              
    def get_interval(self):
        return self.intervals
    
    def get_train_data(self):
        return self.batched_train_data
    
    def get_validation_data(self):
        return self.batched_val_data
    
    def get_test_data(self):
        return self.batched_test_data
    
    def get_data_type(self):
        return self.data_type
    
    def get_name(self):
        return self.name
    
    def get_dataset_size(self):
        return self.dataset_size
    
    def get_data_shape(self):
        return self.input_output_shape

    def get_batch_size(self):
        return self.batch_size

    def get_data(self):
        return self.batched_train_data, self.batched_val_data, self.batched_test_data