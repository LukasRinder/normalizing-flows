from normalizingflows.flow_catalog import PlanarFlow, Made, RealNVP, BatchNorm, NeuralSplineFlow, get_trainable_variables
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from enum import Enum
from utils.types import DataType, FlowType

class Flow():
    def __init__(self, flow_name, num_layers, input_output_shape_tuple, data_type: DataType, hidden_units=None, intervals=None, number_of_bins=None):
        self.flow_type = FlowType[flow_name.lower()]
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        input_dim = input_output_shape_tuple[0]
        output_shape = input_output_shape_tuple[1]
        #assertions
        assert input_dim is not None and isinstance(input_dim, int),\
            "you should at least provide the input_shape and it must have ndims 1 (vector)"

        if self.flow_type is not FlowType.planar:
            assert hidden_units is not None, f"you should provide hidden units for {self.flow_type.name}"
        if self.flow_type is FlowType.neuralspline:
            assert number_of_bins is not None, f"you should provide the number of bins for {self.flow_type.name}"
            assert intervals is not None, f"you should provide the intervals for {self.flow_type.name}"
        if data_type is DataType.mnist or data_type is DataType.celeb:
            assert output_shape is not None, f"you should provide the output dim for {data_type.name}"

        base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(shape=input_dim, dtype=tf.float32))
        permutation = tf.cast(np.concatenate((np.arange(input_dim / 2, input_dim), np.arange(0, input_dim / 2))),
                              tf.int32)
        bijectors = []

        if self.flow_type is FlowType.planar:
            assert data_type is DataType.toydata or data_type is DataType.uci, f"{self.flow_type} is not defined on this type dataset"
            for i in range(0, num_layers):
                bijectors.append(PlanarFlow(input_dimensions=input_dim, case="density_estimation"))

        elif self.flow_type is FlowType.realnvp:
            for i in range(num_layers):
                if data_type is not DataType.toydata:
                    bijectors.append(tfb.BatchNormalization())
                bijectors.append(RealNVP(input_shape=input_dim, n_hidden=hidden_units))
                bijectors.append(tfp.bijectors.Permute(permutation))

            if data_type is DataType.mnist or data_type is DataType.celeb:
                # reshape array to image shape, before: (size*size,)
                bijectors.append(tfb.Reshape(event_shape_out=output_shape,
                                             event_shape_in=(input_dim,)))

        elif self.flow_type is FlowType.neuralspline:
            assert data_type is DataType.toydata or data_type is DataType.uci, f"{flow_name} is not defined on this type dataset {data_type.name}"
            for i in range(num_layers):
                bijectors.append(
                    NeuralSplineFlow(input_dim=input_dim, d_dim=int(input_dim / 2) + 1, number_of_bins=number_of_bins,
                                     b_interval=intervals))
                bijectors.append(tfp.bijectors.Permute(permutation))
                if data_type is not DataType.toydata:
                    bijectors.append(tfb.BatchNormalization())

        elif self.flow_type is FlowType.maf:
            if data_type is not DataType.toydata:
                bijectors.append(BatchNorm(eps=10e-5, decay=0.95))

            for i in range(0, num_layers):
                bijectors.append(tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=Made(params=2, hidden_units=hidden_units, activation="relu")))
                bijectors.append(tfb.Permute(permutation=permutation))

                # add BatchNorm every two layers
                if (i + 1) % int(2) == 0:
                    bijectors.append(BatchNorm(eps=10e-5, decay=0.95))

            if data_type is DataType.mnist or data_type is DataType.celeb:
                bijectors.append(tfb.Reshape(event_shape_out=(output_shape), event_shape_in=(input_dim,)))

        else:
            raise NameError("the flow you chose is not defined")


        bijector = tfb.Chain(bijectors=list(reversed(bijectors)), name=f'chain_of_{self.flow_type.name}')
        dist = tfd.TransformedDistribution(
            distribution=base_dist,
            bijector=bijector,
            #event_shape=[2]  TODO: ask lukas if this is needed?
        )
        self.distribution = dist

    def get_num_layers(self):
        return self.num_layers
    def get_distribution(self):
        return self.distribution
    def get_n_trainable_variables(self):
        return get_trainable_variables(self.distribution)
    def get_flow_type(self):
        return self.flow_type
    def get_flow_shape(self):
        if self.flow_type is FlowType.planar:
            return "no_shape"
        return self.hidden_units 
        
