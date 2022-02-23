## neqkir
# GRU-based char-RNN for text generation
# learning to surprise

# V4

# modifying GRU state initialization
# zero-initialization occurring not once per batch but once per epoch

# Normally LSTM state is cleared at the end of each batch in Keras, but we can control
# it by making the LSTM stateful and calling model.reset_state() to manage this state manually.

# With a custom callback, we reset states at the end of each epoch, which stateful is set to True

# > On top of that we now make states trainable parameters of the model
# > to this end we subclass the GRU layer

# > At the end, we compute some score
# > we score the overlap of generated text with the input vocabulary

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import os
import time

##### TRAINING PARAMETERS
EPOCHS=2 # we will early stop anyway
seq_length = 100
BATCH_SIZE = 32
embedding_dim = 512
rnn_units = 1024
dropout = 0.4

##### INFERENCE PARAMETERS
temperature = 0.7


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

class CustomGRULayer(tf.keras.layers.Layer):
  def __init__(self, rnn_units, batch_size):
    super(CustomGRULayer, self).__init__()
    self.rnn_units = rnn_units
    self.batch_size = batch_size
    self.gru = tf.keras.layers.GRU(self.rnn_units,
                                   stateful=True,
                                   return_sequences=True,
                                   return_state=True,
                                   reset_after=True,
                                   activation='tanh',
                                   recurrent_activation='sigmoid',  
                                   recurrent_dropout=0.2,
                                   dropout=dropout 
                                   )
    self.w=None

  def build(self, input_shape):
    w_init = tf.random_normal_initializer(mean=0.0, stddev=0.2)
    self.w = tf.Variable(
        initial_value=w_init(shape=(self.batch_size, self.rnn_units),
                             dtype='float32'), trainable=True)
    
  def call(self, inputs): 
    return self.gru(inputs, initial_state = self.w)
  

class LearningToSurpriseModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units, batch_size):
    super().__init__(self)

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru_layer = CustomGRULayer(rnn_units = rnn_units, batch_size = batch_size)   
    self.dense = tf.keras.layers.Dense(vocab_size)
 
  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)

    if states is None:
      states = self.gru_layer.gru.get_initial_state(x)

    x, states = self.gru_layer.gru(x, initial_state=states, training=training)
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
    
model = LearningToSurpriseModel(
    # Be sure the vocabulary size matches the `StringLookup` layers.
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE
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

# defining a custom callback for resetting states at the end of period only

gru_layer = model.layers[1]

class CustomCallback(tf.keras.callbacks.Callback):
   def __init__(self, gru_layer):
        self.gru_layer = gru_layer
   def on_epoch_end(self, epoch, logs=None):
        self.gru_layer.gru.reset_states(self.gru_layer.w)
        
model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, callbacks = [EarlyS, CustomCallback(gru_layer)], verbose=1)

# save weights

model.save_weights("learning-to-surprise-weights.h5")

###########################

## GENERATOR

###########################

# same as the training model, but we work on a different batch size
# so we rebuild the model and load training weights

one_step_model = LearningToSurpriseModel(
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE
    )

states = None
# next_char = tf.constant(['La joie '])
next_char = tf.constant(['Au gré, selon la disposition, plénitude, vacuité. Ton acte toujours s’applique à du papier. \
Le vieux Mélodrame occupant la scène, conjointement à la Danse et sous la régie aussi du poëte, satisfait à cette loi. Apitoyé,'])
result = [next_char]

# loading training weights
one_step_model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
              )

input_chars = tf.strings.unicode_split(next_char, 'UTF-8')
input_ids = ids_from_chars(input_chars).to_tensor()

logits = one_step_model(input_ids,states=None)

one_step_model.load_weights("learning-to-surprise-weights.h5")

# Create a mask to prevent "[UNK]" from being generated.
skip_ids = ids_from_chars(['[UNK]'])[:, None]
sparse_mask = tf.SparseTensor(
  # Put a -inf at each bad index.
  values=[-float('inf')]*len(skip_ids),
  indices=skip_ids,
  # Match the shape to the vocabulary
  dense_shape=[len(ids_from_chars.get_vocabulary())])
prediction_mask = tf.sparse.to_dense(sparse_mask)

@tf.function
def generate_one_step(model, inputs, states=None):
  # Convert strings to token IDs.
  input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
  input_ids = ids_from_chars(input_chars).to_tensor()

  # Run the model.
  # predicted_logits.shape is [batch, char, next_char_logits]
  predicted_logits, states = model(inputs=input_ids, states=states,
                                        return_state=True)
  # Only use the last prediction.
  predicted_logits = predicted_logits[:, -1, :]
  predicted_logits = predicted_logits/temperature
  # Apply the prediction mask: prevent "[UNK]" from being generated.
  predicted_logits = predicted_logits + prediction_mask

  # Sample the output logits to generate token IDs.
  predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
  predicted_ids = tf.squeeze(predicted_ids, axis=-1)

  # Convert from token ids to characters
  predicted_chars = chars_from_ids(predicted_ids)

  # Return the characters and model state.
  return predicted_chars, states

# text generation
start = time.time()
for n in range(10000):
    next_char, states = generate_one_step(one_step_model, next_char, states=states)    
    result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)


##############################################
### goodness 

##### Read and word-tokenize the input train text

t_tokenizer = Tokenizer()
t_tokenizer.fit_on_texts([text])#Builds the word index
t_word_index=t_tokenizer.word_index 

t_vocab_size = len(t_word_index) + 1

print ("Training vocabulary: "+str(t_vocab_size)+" words.")



##### Read and word-tokenize the generated text

generated_text=result[0].numpy().decode('utf-8')
g_tokenizer = Tokenizer()
g_tokenizer.fit_on_texts([generated_text])#Builds the word index
g_word_counts=g_tokenizer.word_counts 

g_vocab_size = len(g_word_counts) + 1

print ("Generated vocabulary: "+str(g_vocab_size)+" words.")

g_word_counts=dict(g_word_counts)

diff = set(g_word_counts.keys()).intersection(set(t_word_index.keys()))
num_words_g = sum(list(g_word_counts.values()))
num_words_g_in_t = sum(list(map(g_word_counts.get, list(diff))))
perc_words_g_in_t = 100 * num_words_g_in_t/ num_words_g

perc_words_g_in_t = "{:.2f}".format(perc_words_g_in_t)
output_str=perc_words_g_in_t+" % of all generated words are in the training vocabulary." 

print ( output_str )

with open('mallarme-like.txt','w+') as f:
  f.write("embedding dim: " + str(embedding_dim)+"\n")
  f.write("batch size: " + str(BATCH_SIZE)+"\n")
  f.write("rnn units: " + str(rnn_units)+"\n")
  f.write("temperature: " + str(temperature)+"\n")
  f.write("dropout: " + str(dropout)+"\n")
  f.write("sequence length: " + str(seq_length)+"\n")
  f.write("epochs: " + str(EPOCHS)+"\n\n"+'_'*80+"\n")
  f.write(output_str + '\n\n' + '_'*80+"\n")
  f.write(generated_text + '\n\n' + '_'*80)
  f.write('\nRun time:%f'  %(end - start))


