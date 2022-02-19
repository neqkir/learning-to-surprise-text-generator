#### implements early stopping # accuracy metrics

# custom training implementation

# 30 EPOCHS with early stopping

# starting character Q

# writing result to a file

##There are two variants of the GRU implementation. The default one is based on v3 and
##has reset gate applied to hidden
##state before matrix multiplication. The other one is based on original and has the order reversed.
##The second variant is compatible with CuDNNGRU (GPU-only) and allows inference on CPU.
##Thus it has separate biases for kernel and recurrent_kernel. To use this variant,
##set 'reset_after'=True and recurrent_activation='sigmoid'.

# --> added gpu options to test and make sure right gpu api is used

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

##############################

#### TRAIN

##############################


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

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024


class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
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

history = model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, callbacks = [EarlyS], verbose=1)


###########################

## GENERATOR

###########################


class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
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

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states


one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

start = time.time()
states = None
next_char = tf.constant(['La joie '])
result = [next_char]

for n in range(10000):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)

with open('mallarme-like.txt','a') as f:
  f.write(result[0].numpy().decode('utf-8') + '\n\n' + '_'*80)
  f.write('\nRun time:%f'  %(end - start))

# If you want the model to generate text faster
# the easiest thing you can do is batch the text generation. In the example below the model
# generates 5 outputs in about the same time it took to generate 1 above.

start = time.time()
states = None
next_char = tf.constant(['Sous l’hyacinthe ', 'Mes bouquins ', 'Quel sépulcral ', 'M’introduire chez ', 'Le rat '])
result = [next_char]

for n in range(10000):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
end = time.time()

with open('mallarme-like-x5.txt','a') as f:
  
  for i in range(0,4):
      
    f.write(result[0].numpy().decode('utf-8') + '\n\n' + '_'*80)

    print(result[i].numpy().decode('utf-8'), '\n\n' + '_'*80)

  f.write('\nRun time:%f' % (end - start))

print('\nRun time:%f' % (end - start))

# save generator model

if not os.path.exists('saved_model'):
    os.makedirs('saved_model')
    
tf.saved_model.save(one_step_model, 'saved_model/one_step_gen')

# print metrics

import matplotlib.pyplot as plt

print(history.history.keys())

acc=history.history['sparse_categorical_accuracy']
val_acc=history.history['val_sparse_categorical_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1, len(acc)+1)

plt.plot(epochs,acc,'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('Accuracy.pdf')

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('Loss.pdf')
