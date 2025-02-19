# Copyright 2021, Google LLC. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run script for distributed mean estimation."""

import os
import pprint

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import tensorflow as tf
import tensorflow_privacy as tfp

from distributed_dp import accounting_utils
from distributed_dp import ddpquery_utils
from distributed_dp import dme_utils

flags.DEFINE_boolean('show_plot', False, 'Whether to plot the results.')
flags.DEFINE_boolean('print_output', False, 'Whether to print the outputs.')
flags.DEFINE_integer(
    'run_id', 1, 'ID of the run, useful for identifying '
    'the run when parallelizing this script.')
flags.DEFINE_integer('repeat', 5, 'Number of times to repeat (sequentially).')
flags.DEFINE_string('output_dir', '/tmp/ddp_dme_outputs', 'Output directory.')
flags.DEFINE_string('tag', '', 'Extra subfolder for the output result files.')
flags.DEFINE_enum('mechanism', 'ddgauss', ['ddgauss'], 'DDP mechanism to use.')
flags.DEFINE_float('norm', 10.0, 'Norm of the randomly generated vectors.')
flags.DEFINE_integer(
    'k_stddevs', 2, 'Number of standard deviations of the '
    'noised, quantized, aggregated siginal to bound.')
flags.DEFINE_boolean(
    'sqrtn_norm_growth', False, 'Whether to assume the bound '
    'norm(sum_i x_i) <= sqrt(n) * c.')

FLAGS = flags.FLAGS


def experiment(bits,
               clip,
               beta,
               client_data,
               epsilons,
               delta,
               mechanism,
               k_stddevs=2,
               sqrtn_norm_growth=False):
  """Run a distributed mean estimation experiment.

  Args:
    bits: A list of compression bits to use.
    clip: The initial L2 norm clip.
    beta: A hyperparameter controlling the concentration inequality for the
      probabilistic norm bound after randomized rounding.
    client_data: A Python list of `n` np.array vectors, each with shape (d,).
    epsilons: A list of target epsilon values for comparison (serve as x-axis).
    delta: The delta for approximate DP.
    mechanism: A string specifying the mechanism to compare against Gaussian.
    k_stddevs: The number of standard deviations to keep for modular clipping.
      Defaults to 2.
    sqrtn_norm_growth: Whether to assume the norm of the sum of the vectors grow
      at a rate of `sqrt(n)` (i.e. norm(sum_i x_i) <= sqrt(n) * c). If `False`,
      we use the upper bound `norm(sum_i x_i) <= n * c`.

  Returns:
    Experiment results as lists of MSE.
  """

  def mse(a, b):
    assert a.shape == b.shape
    return np.square(a - b).mean()

  # Initial fixed params.
  num_clients = len(client_data)
  d = len(client_data[0])
  padded_dim = np.math.pow(2, np.ceil(np.log2(d)))
  client_template = tf.zeros_like(client_data[0])

  # `client_data` has shape (n, d).
  true_avg_vector = np.mean(client_data, axis=0)

  # 1. Baseline: central continuous Gaussian.
  gauss_mse_list = []
  for eps in epsilons:
    # Analytic Gaussian.
    gauss_stddev = accounting_utils.analytic_gauss_stddev(eps, delta, clip)
    gauss_query = tfp.GaussianSumQuery(l2_norm_clip=clip, stddev=gauss_stddev)
    gauss_avg_vector = dme_utils.compute_dp_average(
        client_data, gauss_query, is_compressed=False, bits=None)
    gauss_mse_list.append(mse(gauss_avg_vector, true_avg_vector))

  # 2. Distributed DP: try each `b` separately.
  ddp_mse_list_per_bit = []
  for bit in bits:
    discrete_mse_list = []
    for eps in epsilons:
      if mechanism == 'ddgauss':
        gamma, local_stddev = accounting_utils.ddgauss_params(
            q=1,
            epsilon=eps,
            l2_clip_norm=clip,
            bits=bit,
            num_clients=num_clients,
            dim=padded_dim,
            delta=delta,
            beta=beta,
            steps=1,
            k=k_stddevs,
            sqrtn_norm_growth=sqrtn_norm_growth)
        scale = 1.0 / gamma
      else:
        raise ValueError(f'Unsupported mechanism: {mechanism}')

      ddp_query = ddpquery_utils.build_ddp_query(
          mechanism,
          local_stddev,
          l2_norm_bound=clip,
          beta=beta,
          padded_dim=padded_dim,
          scale=scale,
          client_template=client_template)

      distributed_avg_vector = dme_utils.compute_dp_average(
          client_data, ddp_query, is_compressed=True, bits=bit)
      discrete_mse_list.append(mse(distributed_avg_vector, true_avg_vector))

    ddp_mse_list_per_bit.append(discrete_mse_list)

  # Convert to np arrays and do some checks
  gauss_mse_list = np.array(gauss_mse_list)
  ddp_mse_list_per_bit = np.array(ddp_mse_list_per_bit)

  assert gauss_mse_list.shape == (len(epsilons),)
  assert ddp_mse_list_per_bit.shape == (len(bits), len(epsilons))

  return gauss_mse_list, ddp_mse_list_per_bit


def experiment_repeated(bits,
                        clip,
                        beta,
                        client_data_list,
                        repeat,
                        epsilons,
                        delta,
                        mechanism,
                        k_stddevs=2,
                        sqrtn_norm_growth=False):
  """Sequentially repeat the experiment (see `experiment()` for parameters)."""
  assert len(client_data_list) == repeat
  n, d = len(client_data_list[0]), len(client_data_list[0][0])
  print(f'Sequentially repeating the experiment {len(client_data_list)} times '
        f'for n={n}, d={d}, mechanism={mechanism}, c={clip}, bits={bits}, beta='
        f'{beta:.3f}, eps={epsilons}, k={k_stddevs}, sng={sqrtn_norm_growth}')

  repeat_results = []
  for client_data in client_data_list:
    repeat_results.append(
        experiment(
            bits=bits,
            clip=clip,
            beta=beta,
            client_data=client_data,
            epsilons=epsilons,
            delta=delta,
            mechanism=mechanism,
            k_stddevs=k_stddevs,
            sqrtn_norm_growth=sqrtn_norm_growth))

  repeat_gauss_mse_list, repeat_ddp_mse_list_per_bit = zip(*repeat_results)

  repeat_gauss_mse_list = np.array(repeat_gauss_mse_list)
  repeat_ddp_mse_list_per_bit = np.array(repeat_ddp_mse_list_per_bit)

  assert len(repeat_results) == repeat
  assert repeat_gauss_mse_list.shape == (repeat, len(epsilons))
  assert (repeat_ddp_mse_list_per_bit.shape == (repeat, len(bits),
                                                len(epsilons)))

  return repeat_gauss_mse_list, repeat_ddp_mse_list_per_bit


def mean_confidence_interval(data, confidence=0.95):
  # `data` should have shape (repeat, len(x-axis)).
  n = len(data)
  m, se = np.mean(data, axis=0), scipy.stats.sem(data, axis=0)
  h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
  return m, m - h, m + h


def plot_curve(subplot, epsilons, data, label):
  assert len(data.shape) == 2, 'data should be (repeat, len(x-axis))'
  means, lower, upper = mean_confidence_interval(data)
  subplot.plot(epsilons, means, label=label)
  subplot.fill_between(epsilons, lower, upper, alpha=0.2, edgecolor='face')


def main(_):
  """Run distributed mean estimation experiments."""
  clip = FLAGS.norm
  delta = 1e-5
  use_log = True  # Whether to use log-scale for y-axis.
  k_stddevs = FLAGS.k_stddevs
  sqrtn_norm_growth = FLAGS.sqrtn_norm_growth
  repeat = FLAGS.repeat

  # Parallel subplots for different n=num_clients and d=dimension.
  nd_zip = [(100, 250), (1000, 250)]
  # nd_zip = [(10000, 2000)]

  # Curves within a subplot.
  bits = [10, 12, 14, 16]
  # bits = [14, 16, 18, 20]

  # X-axis: epsilons.
  epsilons = [0.75] + list(np.arange(1, 6.5, 0.5))

  _, ax = plt.subplots(1, max(2, len(nd_zip)), figsize=(20, 5))

  results = []
  for j, (n, d) in enumerate(nd_zip):
    client_data_list = [
        dme_utils.generate_client_data(d, n, l2_norm=clip)
        for _ in range(repeat)
    ]
    beta = np.exp(-0.5)

    # Run experiment with repetition.
    rep_gauss_mse_list, rep_ddp_mse_list_per_bit = experiment_repeated(
        bits,
        clip,
        beta,
        client_data_list,
        repeat,
        epsilons,
        delta,
        mechanism=FLAGS.mechanism,
        k_stddevs=k_stddevs,
        sqrtn_norm_growth=sqrtn_norm_growth)

    # Generate some basic plots here. Use the saved results to generate plots
    # with custom style if needed.
    if FLAGS.show_plot:
      subplot = ax[j]
      # Continuous Gaussian.
      plot_curve(
          subplot, epsilons, rep_gauss_mse_list, label='Continuous Gaussian')
      # Distributed DP.
      for index, bit in enumerate(bits):
        plot_curve(
            subplot,
            epsilons,
            rep_ddp_mse_list_per_bit[:, index],
            label=f'{FLAGS.mechanism} (B = {bit})')

      subplot.set(xlabel='Epsilon', ylabel='MSE')
      subplot.set_title(f'(n={n}, d={d}, k={k_stddevs})')
      subplot.set_yscale('log' if use_log else 'linear')
      subplot.legend()

    result_dic = {
        'n': n,
        'd': d,
        'rep': repeat,
        'c': clip,
        'bits': bits,
        'k_stddevs': k_stddevs,
        'epsilons': epsilons,
        'mechanism': FLAGS.mechanism,
        'sqrtn_norm_growth': sqrtn_norm_growth,
        'gauss': rep_gauss_mse_list,
        FLAGS.mechanism: rep_ddp_mse_list_per_bit,
    }
    results.append(result_dic)

    if FLAGS.print_output:
      print(f'n={n}, d={d}:')
      pprint.pprint(result_dic)

  # Save to file.
  fname = f'rp={repeat},rid={FLAGS.run_id}.txt'
  fname = fname.replace(' ', '')
  result_str = pprint.pformat(results)
  dirname = os.path.join(FLAGS.output_dir, FLAGS.tag)
  if not os.path.exists(dirname):
    os.makedirs(dirname)
  out_path = os.path.join(dirname, fname)
  with open(out_path, 'w') as f:
    f.write(result_str)
  print('Results saved to', out_path)

  if FLAGS.print_output:
    print('*' * 80)
    print(fname)
    print('*' * 10 + 'Results (copy and `eval()` in Python):')
    print(result_str)
    print('*' * 80)
    print('Copy the above results and `eval()` them as a string in Python.')

  if FLAGS.show_plot:
    plt.show()

  print(f'Run {FLAGS.run_id} done.')


if __name__ == '__main__':
  app.run(main)
