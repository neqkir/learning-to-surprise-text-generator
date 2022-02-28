## neqkir
#
# Seq2seq for text generation
#
# using TensorFlow add-ons and based on models for translation
#
#

# pip install tensorflow-addons==0.11.2

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import unicodedata
import re
import numpy as np
import os
import io
import time

######## PARAMETERS

FILE_PATH = "mallarme.txt"
BUFFER_SIZE = 32000
BATCH_SIZE = 64
# Let's limit the #training examples for faster training
seq_length = 100
embedding_dim = 256
units = 1024
EPOCHS = 10


######## DATA

# Read, then decode for py2 compat.
text = open(FILE_PATH, 'rb').read().decode(encoding='utf-8')
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

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

len_data=len(list(dataset))
validation_dataset = dataset.take(int(len_data*.2))
train_dataset = dataset.skip(int(len_data*.2))

# Length of the vocabulary in chars
vocab_size = len(vocab)+1

example_input_batch, example_target_batch = next(iter(train_dataset))
print(example_input_batch.shape)
print(example_target_batch.shape)  

examples_per_epoch = len(text)//(seq_length+1)

######### FOR TRAINING

## Encoder stack

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    ##-------- LSTM layer in Encoder ------- ##
    self.lstm_layer = tf.keras.layers.LSTM(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, h, c = self.lstm_layer(x, initial_state = hidden)
    return output, h, c

  def initialize_hidden_state(self):
    return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))]

# Test Encoder Stack

encoder = Encoder(vocab_size, embedding_dim, units, BATCH_SIZE)

sample_hidden = encoder.initialize_hidden_state()# sample input
sample_output, sample_h, sample_c = encoder(example_input_batch, sample_hidden)# sample output
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder h vecotr shape: (batch size, units) {}'.format(sample_h.shape))
print ('Encoder c vector shape: (batch size, units) {}'.format(sample_c.shape))

## Decoder stack

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, attention_type='luong'):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.attention_type = attention_type

    # Embedding Layer
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    #Final Dense layer on which softmax will be applied
    self.fc = tf.keras.layers.Dense(vocab_size)

    # Define the fundamental cell for decoder recurrent structure
    self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)

    # Sampler
    self.sampler = tfa.seq2seq.sampler.TrainingSampler()

    # Create attention mechanism with memory = None
    self.attention_mechanism = self.build_attention_mechanism(self.dec_units, 
                                                              None, self.batch_sz*[seq_length], self.attention_type)

    # Wrap attention mechanism with the fundamental rnn cell of decoder
    self.rnn_cell = self.build_rnn_cell(batch_sz)

    # Define the decoder with respect to fundamental rnn cell
    self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc)

  def build_rnn_cell(self, batch_sz):
    rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnn_cell, 
                                  self.attention_mechanism, attention_layer_size=self.dec_units)
    return rnn_cell

  def build_attention_mechanism(self, dec_units, memory, memory_sequence_length, attention_type='luong'):
    # ------------- #
    # typ: Which sort of attention (Bahdanau, Luong)
    # dec_units: final dimension of attention outputs 
    # memory: encoder hidden states of shape (batch_size, seq_length, enc_units)
    # memory_sequence_length: 1d array of shape (batch_size) with every element set to seq_length (for masking purpose)

    if(attention_type=='bahdanau'):
      return tfa.seq2seq.BahdanauAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)
    else:
      return tfa.seq2seq.LuongAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)

  def build_initial_state(self, batch_sz, encoder_state, Dtype):
    # setup initial states with the encoder's states
    decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_sz, dtype=Dtype)
    decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
    return decoder_initial_state

  def call(self, inputs, initial_state):
    x = self.embedding(inputs)
    outputs, _, _ = self.decoder(x, initial_state=initial_state, sequence_length=self.batch_sz*[seq_length-1])
    return outputs

# Test decoder stack

decoder = Decoder(vocab_size, embedding_dim, units, BATCH_SIZE, 'luong')

sample_x = tf.random.uniform((BATCH_SIZE, seq_length))
decoder.attention_mechanism.setup_memory(sample_output) # set encoder's output in the attention mechanism
initial_state = decoder.build_initial_state(BATCH_SIZE, [sample_h, sample_c], tf.float32) # set encoder's states

sample_decoder_outputs = decoder(sample_x, initial_state)

print("Decoder Outputs Shape: ", sample_decoder_outputs.rnn_output.shape)

# Defining optimizer and loss function

optimizer = tf.keras.optimizers.Adam()

def loss_function(real, pred):
  # real shape = (BATCH_SIZE, seq_length)
  # pred shape = (BATCH_SIZE, seq_length, tar_vocab_size )
  cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
  loss = cross_entropy(y_true=real, y_pred=pred)
  mask = tf.logical_not(tf.math.equal(real,0))   #output 0 for y=0 else output 1
  mask = tf.cast(mask, dtype=loss.dtype)  
  loss = mask* loss
  loss = tf.reduce_mean(loss)
  return loss

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_h, enc_c = encoder(inp, enc_hidden)

    dec_input = targ[ : , :-1 ] # Ignore <end> token
    real = targ[ : , 1: ]         # ignore <start> token

    # Set the AttentionMechanism object with encoder_outputs
    decoder.attention_mechanism.setup_memory(enc_output)

    # Create AttentionWrapperState as initial_state for decoder
    decoder_initial_state = decoder.build_initial_state(BATCH_SIZE, [enc_h, enc_c], tf.float32)
    pred = decoder(dec_input, decoder_initial_state)
    logits = pred.rnn_output
    loss = loss_function(real, logits)

  variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))

  return loss


###### TRAINING

for epoch in range(EPOCHS):
  start = time.time()

  enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0
  # print(enc_hidden[0].shape, enc_hidden[1].shape)

  for (batch, (inp, targ)) in enumerate(train_dataset.take(examples_per_epoch)):
    batch_loss = train_step(inp, targ, enc_hidden)
    total_loss += batch_loss

    if batch % 100 == 0:
      print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))
  # saving (checkpoint) the model every 2 epochs
  if (epoch + 1) % 2 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)

  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / examples_per_epoch))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

############ FOR GENERATION

def generate_one_step(sentence, temperature=1.0,batch_size=1):

  input_chars = tf.strings.unicode_split(sentence, 'UTF-8')
  input_ids = ids_from_chars(input_chars).to_tensor()

  inference_batch_size = batch_size
  result = ''

  enc_start_state = [tf.zeros((inference_batch_size, units)), tf.zeros((inference_batch_size,units))]
  enc_out, enc_h, enc_c = encoder(inputs, enc_start_state)

  dec_h = enc_h
  dec_c = enc_c

  start_tokens = tf.fill([inference_batch_size], list(input_chars)[0])
  end_token = list(input_chars)[-1]

  greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

  # Instantiate BasicDecoder object
  decoder_instance = tfa.seq2seq.BasicDecoder(cell=decoder.rnn_cell, sampler=greedy_sampler, output_layer=decoder.fc)
  # Setup Memory in decoder stack
  decoder.attention_mechanism.setup_memory(enc_out)

  # set decoder_initial_state
  decoder_initial_state = decoder.build_initial_state(inference_batch_size, [enc_h, enc_c], tf.float32)

  ### Since the BasicDecoder wraps around Decoder's rnn cell only, you have to ensure that the inputs to BasicDecoder 
  ### decoding step is output of embedding layer. tfa.seq2seq.GreedyEmbeddingSampler() takes care of this. 
  ### You only need to get the weights of embedding layer, which can be done by decoder.embedding.variables[0]
  ### and pass this callabble to BasicDecoder's call() function

  decoder_embedding_matrix = decoder.embedding.variables[0]

  predicted_logits, _, _ = decoder_instance(decoder_embedding_matrix, start_tokens = start_tokens, end_token= end_token, initial_state=decoder_initial_state)

  predicted_logits = predicted_logits[:, -1, :]
  predicted_logits = predicted_logits/temperature
  
  # Sample the output logits to generate token IDs.
  predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
  predicted_ids = tf.squeeze(predicted_ids, axis=-1)
  
  ####  --> check if needed predicted_ids = predicted_logits.sample_id.numpy()
  
  # Convert from token ids to characters
  predicted_chars = chars_from_ids(predicted_ids)

  return predicted_chars

next_char = tf.constant(['La joie '])

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

start = time.time()
for n in range(1000):
    next_char, states = generate_one_step(next_char, temperature=0.8)    
    result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)


#### ---> check if needed

### Create a mask to prevent "[UNK]" from being generated.
##skip_ids = ids_from_chars(['[UNK]'])[:, None]
##sparse_mask = tf.SparseTensor(values=[-float('inf')]*len(skip_ids),# Put a -inf at each bad index.
##                              indices=skip_ids,# Match the shape to the vocabulary
##                              dense_shape=[len(ids_from_chars.get_vocabulary())]
##                              )
##prediction_mask = tf.sparse.to_dense(sparse_mask)
