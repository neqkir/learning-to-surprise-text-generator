## neqkir
# GRU-based char-RNN for text generation
# learning to surprise

# modifying GRU state initialization
# zero-initialization occurring not once per batch but once per epoch

# Normally LSTM state is cleared at the end of each batch in Keras, but we can control
# it by making the LSTM stateful and calling model.reset_state() to manage this state manually.

# > build a new model for inference, extending the training model
# > loading the training weights into the model

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import os
import time

EPOCHS=30 # we will early stop anyway

#####################

##   DATA

#####################

# Read, then decode for py2 compat.
text = open("mallarme.txt", 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print(f'Length of text: {len(text)} characters')

# Take a look at the first 1000 characters in text
print(text[:1000])

# The unique characters in the file
vocab = sorted(set(text))
print(f'{len(vocab)} unique characters')

ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab), mask_token=None)

chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

# For training you'll need a dataset of (input, label) pairs.
# Where input and label are sequences.
# At each time step the input is the current character and the label is the next character.
# Here's a function that takes a sequence as input, duplicates, and shifts it to align the input
# and label for each timestep:

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)


# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).

BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

len_data=len(list(dataset))
validation_dataset = dataset.take(int(len_data*.2))
train_dataset = dataset.skip(int(len_data*.2))

# Length of the vocabulary in chars
vocab_size = len(vocab)

##############################

#### TRAIN

##############################

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024


class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   stateful=True,
                                   return_sequences=True,
                                   return_state=True,
                                   reset_after=True,
                                   activation='tanh',
                                   recurrent_activation='sigmoid',  
                                   recurrent_dropout=0.2,
                                   dropout=0.2  
                                   )
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x

  @tf.function
  def train_step(self, inputs):
    # unpack the data
    inputs, labels = inputs
  
    with tf.GradientTape() as tape:
      predictions = self(inputs, training=True) # forward pass
      # Compute the loss value
      # (the loss function is configured in `compile()`)
      loss=self.compiled_loss(labels, predictions, regularization_losses=self.losses)

    # compute the gradients
    grads=tape.gradient(loss, model.trainable_variables)
    # Update weights
    self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # Update metrics (includes the metric that tracks the loss)
    self.compiled_metrics.update_state(labels, predictions)

    # Return a dict mapping metric names to current value
    return {m.name: m.result() for m in self.metrics}
    
model = MyModel(
    # Be sure the vocabulary size matches the `StringLookup` layers.
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units
    )

# checking the shape of the output
for input_example_batch, target_example_batch in train_dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

model.summary()

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

# checking the mean_loss
example_batch_loss = loss(target_example_batch, example_batch_predictions)
mean_loss = example_batch_loss.numpy().mean()
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("Mean loss:        ", mean_loss)

# A newly initialized model shouldn't be too sure of itself, the output logits should all have similar magnitudes.
# To confirm this you can check that the exponential of the mean loss is approximately equal to the vocabulary size.
# A much higher loss means the model is sure of its wrong answers, and is badly initialized:
tf.exp(mean_loss).numpy()

model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[
                  tf.keras.metrics.SparseCategoricalAccuracy()]
              )

# setting early-stopping
EarlyS = EarlyStopping(monitor = 'val_loss', mode = 'min', restore_best_weights=True, patience=10, verbose = 1)

for i in range(EPOCHS):
  print("EPOCH "+str(i+1)+"/"+str(EPOCHS))
  model.fit(train_dataset, validation_data=validation_dataset, epochs=1, callbacks = [EarlyS], verbose=1)
  model.reset_states()

# save weights

model.save_weights("learning-to-surprise-weights.h5")

###########################

## GENERATOR

###########################


class OneStepModel(tf.keras.Model):
  
  def __init__(self, vocab_size, embedding_dim, rnn_units, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__(self)

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   stateful=True,
                                   return_sequences=True,
                                   return_state=True,
                                   reset_after=True,
                                   activation='tanh',
                                   recurrent_activation='sigmoid',  
                                   recurrent_dropout=0.2,
                                   dropout=0.2  
                                   )
    self.dense = tf.keras.layers.Dense(vocab_size)  

    self.temperature = temperature
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  def call(self, inputs, states=None, return_state=False):
    x = inputs
    x = self.embedding(x, training=False)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=False)
    x = self.dense(x, training=False)

    if return_state:
      return x, states
    else:
      return x

one_step_model = OneStepModel(
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    chars_from_ids=chars_from_ids,
    ids_from_chars=ids_from_chars,
    temperature=1.0
    )

states = None
next_char = tf.constant(['La joie '])
result = [next_char]

# loading training weights
one_step_model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
              )

input_chars = tf.strings.unicode_split(next_char, 'UTF-8')
input_ids = one_step_model.ids_from_chars(input_chars).to_tensor()

logits = one_step_model(input_ids,states=None)

one_step_model.load_weights("learning-to-surprise-weights.h5")

# text generation
start = time.time()
for n in range(10000):

    input_chars = tf.strings.unicode_split(next_char, 'UTF-8')
    input_ids = one_step_model.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = one_step_model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/one_step_model.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + one_step_model.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    next_char = one_step_model.chars_from_ids(predicted_ids)
    
    result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)

with open('mallarme-like.txt','a') as f:
  f.write(result[0].numpy().decode('utf-8') + '\n\n' + '_'*80)
  f.write('\nRun time:%f'  %(end - start))
