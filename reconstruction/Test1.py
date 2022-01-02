import tensorflow_federated as tff
import tensorflow as tf
import collections
import os
from tensorflow_federated.python.simulation import hdf5_client_data

# wrap it as a TFF computation
# <=> foo = tff.tf_computation(lambda x: x > 10, tf.int32)
@tff.tf_computation(tf.int32)
def foo(x):
  return x > 10


# <=> foo = tff.tf_computation(lambda: tf.constant(10))
@tff.tf_computation
def foo():
  return tf.constant(10)

#  <=> foo = tff.tf_computation(lambda x, y: x > y)
@tff.tf_computation
def foo(x, y):
  return x > y



class GlobalEmbedding(tf.keras.layers.Layer):
  """A custom Keras Embedding layer used for the global embeddings.

  The `GlobalEmbedding`s correspond to embeddings with input dimension size
  vocabulary_size + 3 (pad/bos/eos). The updates to these embeddings are sent
  to the server.
  """

  def __init__(
      self,
      total_vocab_size: int,
      embedding_dim: int,
      mask_zero: bool = True,
      initializer: tf.keras.initializers = tf.keras.initializers.random_uniform,
      **kwargs):
    super(GlobalEmbedding, self).__init__(**kwargs)
    self.total_vocab_size = total_vocab_size
    self.embedding_dim = embedding_dim
    self.mask_zero = mask_zero
    self.initializer = initializer

  def build(self, input_shape):
    self.embeddings = self.add_weight(
        shape=(self.total_vocab_size, self.embedding_dim),
        initializer=self.initializer,
        name='global_embedding',
    )

  def call(self, inputs):
    embedding_inputs = tf.where(inputs < self.total_vocab_size, inputs,
                                tf.zeros_like(input=inputs))
    embeddings = tf.nn.embedding_lookup(self.embeddings, embedding_inputs)
    return tf.where(
        tf.expand_dims(inputs < self.total_vocab_size, axis=-1), embeddings,
        tf.zeros_like(input=embeddings))

  def compute_mask(self, inputs, mask=None):
    if not self.mask_zero:
      return None
    return tf.not_equal(inputs, 0)


class LocalEmbedding(tf.keras.layers.Layer):
  """A custom Keras Embedding layer used for the local embeddings.

  The `LocalEmbedding`s correspond to embeddings of input size
  number of out of vocabulary buckets.
  These embeddings are reconstructed locally at the beginning of every round,
  and their updates never leave the device.
  """

  def __init__(
      self,
      input_dim: int,
      embedding_dim: int,
      total_vocab_size: int,
      mask_zero: bool = True,
      initializer: tf.keras.initializers = tf.keras.initializers.random_uniform,
      **kwargs):
    super(LocalEmbedding, self).__init__(**kwargs)
    self.input_dim = input_dim
    self.embedding_dim = embedding_dim
    self.mask_zero = mask_zero
    self.total_vocab_size = total_vocab_size
    self.initializer = initializer

  def build(self, input_shape):
    self.embeddings = self.add_weight(
        shape=(self.input_dim, self.embedding_dim),
        initializer=self.initializer,
        name='local_embedding',
    )

  def call(self, inputs):
    embedding_inputs = tf.where(inputs >= self.total_vocab_size,
                                inputs - self.total_vocab_size,
                                tf.zeros_like(input=inputs))
    embeddings = tf.nn.embedding_lookup(self.embeddings, embedding_inputs)
    return tf.where(
        tf.expand_dims(inputs >= self.total_vocab_size, axis=-1), embeddings,
        tf.zeros_like(input=embeddings))

  def compute_mask(self, inputs, mask=None):
    if not self.mask_zero:
      return None
    return tf.not_equal(inputs, 0)


def create_recurrent_reconstruction_model(
    vocab_size: int = 10000,
    num_oov_buckets: int = 1,
    embedding_size: int = 96,
    latent_size: int = 670,
    num_layers: int = 1,
    input_spec=None,
    global_variables_only: bool = False,
    name: str = 'rnn_recon_embeddings',
):

  if vocab_size < 0:
    raise ValueError('The vocab_size is expected to be greater than, or equal '
                     'to 0. Got {}'.format(vocab_size))

  if num_oov_buckets <= 0:
    raise ValueError('The number of out of vocabulary buckets is expected to '
                     'be greater than 0. Got {}'.format(num_oov_buckets))

  global_layers = []
  local_layers = []

  total_vocab_size = vocab_size + 3  # pad/bos/eos.
  extended_vocab_size = total_vocab_size + num_oov_buckets  # pad/bos/eos + oov.
  inputs = tf.keras.layers.Input(shape=(None,), dtype=tf.int64)

  global_embedding = GlobalEmbedding(
      total_vocab_size=total_vocab_size,
      embedding_dim=embedding_size,
      mask_zero=True,
      name='global_embedding_layer')
  global_layers.append(global_embedding)

  local_embedding = LocalEmbedding(
      input_dim=num_oov_buckets,
      embedding_dim=embedding_size,
      total_vocab_size=total_vocab_size,
      mask_zero=True,
      name='local_embedding_layer')
  local_layers.append(local_embedding)

  projected = tf.keras.layers.Add()(
      [global_embedding(inputs),
       local_embedding(inputs)])

  for i in range(num_layers):
    layer = tf.keras.layers.LSTM(
        latent_size, return_sequences=True, name='lstm_' + str(i))
    global_layers.append(layer)
    processed = layer(projected)
    # A projection changes dimension from rnn_layer_size to
    # input_embedding_size.
    projection = tf.keras.layers.Dense(
        embedding_size, name='projection_' + str(i))
    global_layers.append(projection)
    projected = projection(processed)

  # We predict the OOV tokens as part of the output vocabulary.
  last_layer = tf.keras.layers.Dense(
      extended_vocab_size, activation=None, name='last_layer')
  global_layers.append(last_layer)
  logits = last_layer(projected)

  model = tf.keras.Model(inputs=inputs, outputs=logits, name=name)

  print("##### stackover flow model #####")
  model.summary()

  from tensorflow.keras.utils import plot_model
  plot_model(model, to_file=model.name+".png", show_shapes=True)
  print("##### stackover flow model #####")

  if input_spec is None:
    input_spec = collections.OrderedDict(
        x=tf.TensorSpec(shape=(None,), dtype=tf.int64),
        y=tf.TensorSpec(shape=(None,), dtype=tf.int64))

  # Merge local layers into global layers if needed.
  if global_variables_only:
    global_layers.extend(local_layers)
    local_layers = []

  return


if __name__ =="__main__":
    # train_clientdata, validation_clientdata, test_clientdata = (
    # tff.simulation.datasets.stackoverflow.load_data())

    # foo = tff.tf_computation(lambda x: x > 10, tf.int32)
    # str(foo.type_signature) == '(int32 -> bool)'

    # foo = tff.tf_computation(tf.add, (tf.int32, tf.int32))
    # str(foo.type_signature) == '(<int32,int32> -> int32)'

    # foo = tff.tf_computation(lambda: tf.constant(10))

    # x = tf.Variable(0)
    # y = tf.assign(x, 1)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print(sess.run(x))
    #     print(sess.run(y))
    #     print(sess.run(x))

    # create_recurrent_reconstruction_model()

    path = '/root/.keras/datasets/stackoverflow.tar.bz2'
    dir_path = os.path.dirname(path)
    train_client_data = hdf5_client_data.HDF5ClientData(
        os.path.join(dir_path, 'stackoverflow_train.h5'))
    held_out_client_data = hdf5_client_data.HDF5ClientData(
        os.path.join(dir_path, 'stackoverflow_held_out.h5'))
    test_client_data = hdf5_client_data.HDF5ClientData(
        os.path.join(dir_path, 'stackoverflow_test.h5'))
    pass


