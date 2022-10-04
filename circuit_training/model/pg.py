import functools
import sys
from typing import Optional

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tf_agents.networks import nest_map # https://www.tensorflow.org/agents/api_docs/python/tf_agents/networks/NestMap
from tf_agents.networks import sequential # https://www.tensorflow.org/agents/api_docs/python/tf_agents/networks/Sequential
from tf_agents.typing import types


init = tf.keras.initializers.GlorotUniform(seed=42 % sys.maxsize)

dense = functools.partial(
      tf.keras.layers.Dense, activation='relu', kernel_initializer=init)
fc_layer_units = [64, 64, 64, 64]

def no_op_layer():
    # Wraps arbitrary expressions as a Layer object.
    # x -> x no changes
    return tf.keras.layers.Lambda(lambda x: x)

def projection_layer():
    return tf.keras.layers.Dense(
        128,
        activation=None,
        kernel_initializer=init,
        name='projection_layer')

def create_dist(logits_and_mask):
    # Apply mask onto the logits such that infeasible actions will not be taken.
    logits, mask = logits_and_mask.values()

nm = nest_map.NestMap({
              'graph_embedding':
                  tf.keras.Sequential(
                    #flatten input + fully connect 4 layers + 
                      [tf.keras.layers.Flatten()] +
                      [dense(num_units)
                       for num_units in fc_layer_units] + [projection_layer()]),
              'mask':
                  no_op_layer(),
          })

print(nm.build((1,64)))
print(nm.summary())
