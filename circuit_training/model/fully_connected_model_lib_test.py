# coding=utf-8
# Copyright 2021 The Circuit Training Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper functions for creating the fully connected models.

This model architecture creates a simple agent which can't generealize
over multiple netlists, but it has a low inference and train cost which makes it
more suitable that the GCN-based model for reward function development.
"""

import functools
import os,sys
from typing import Optional

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tf_agents.networks import nest_map
from tf_agents.networks import sequential
from tf_agents.typing import types

from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils

# additional
from absl import app
from absl import flags
from absl import logging
from circuit_training.environment import environment
from circuit_training.learning import agent
from circuit_training.learning import static_feature_cache
from circuit_training.model import model
from circuit_training.utils import test_utils

from circuit_training.model import fully_connected_model_lib
FLAGS = flags.FLAGS

if not FLAGS.plc_wrapper_main:
  flags.DEFINE_string('plc_wrapper_main', 'plc_wrapper_main',
                    'Path to plc_wrapper_main binary.')

flags.DEFINE_string('plc_wrapper_main', 'plc_wrapper_main',
                    'Path to plc_wrapper_main binary.')

FLAGS = flags.FLAGS

_TESTDATA_DIR = ('circuit_training/'
                 'environment/test_data/ariane')

def test_actor_net():
  env = environment.create_circuit_environment(
    netlist_file=os.path.join(
        _TESTDATA_DIR, 'netlist.pb.txt'),
    init_placement=os.path.join(
        _TESTDATA_DIR, 'initial.plc'))
  observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
    spec_utils.get_tensor_specs(env))
  
  return env

def main(argv):
    test_actor_net()

app.run(main)