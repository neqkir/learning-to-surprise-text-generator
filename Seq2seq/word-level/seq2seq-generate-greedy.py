## neqkir
#
# Seq2seq for text generation
#
# Using TensorFlow add-ons and based on models for translation
#
# Tokenizing at word level
#
# Generator --> loading best weights from checkpoint

# pip install tensorflow-addons==0.11.2

import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split

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
max_words=10
embedding_dim = 256
units = 1024
EPOCHS = 60


######## DATA

class Seq2seqTextGenDataset:
    def __init__(self):
        self.inp_tokenizer = None
        self.targ_tokenizer = None
        self.num_examples = 0

    def unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def preprocess_text(self, w):
        w = self.unicode_to_ascii(w.lower().strip())

        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)

        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z?.!,¿']+", " ", w)

        w = w.strip()

        return w

    def add_start_end_tok(self, w):
        
        return '<start> ' + ' '.join(w) + ' <end>'

    def split_input_target(self, text, words_per_sequ):

        text = text.split()
        text=text[:(len(text)//words_per_sequ)*words_per_sequ]
        text_sequences = [text[i:i+words_per_sequ] for i in range(0, len(text), words_per_sequ)]

        return [[self.add_start_end_tok(w[1:]), self.add_start_end_tok(w[:-1])] for w in text_sequences]

    def create_dataset(self, path):
        text = open(path, 'rb').read().decode(encoding='utf-8')
        print(text[:1000])

        text = self.preprocess_text( text )

        # Split text into sequences of characters
        sequ_pairs = self.split_input_target(text, max_words)

        # print(sequ_pairs[:5])

        return zip(*sequ_pairs)

    def tokenize(self, sequences):
        # tokenizes input or target sequences
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>')
        tokenizer.fit_on_texts(sequences)
        
        ## tf.keras.preprocessing.text.Tokenizer.texts_to_sequences converts string (w1, w2, w3, ......, wn) 
        ## to a list of correspoding integer ids of words (id_w1, id_w2, id_w3, ...., id_wn)
        tensor = tokenizer.texts_to_sequences(sequences) 

        ## tf.keras.preprocessing.sequence.pad_sequences takes argument a list of integer id sequences 
        ## and pads the sequences to match the longest sequences in the given input
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

        return tensor, tokenizer

    def load_dataset(self, path):
        # creating cleaned input, output pairs
        targ_sequences, inp_sequences = self.create_dataset(path)

        for i in range(3):
            print (str(i)+"/n"+str(targ_sequences[i])+"/n"+str(inp_sequences[i]))

        # -->
        #0
        #<start> action pour les autres editions de ce texte , <end>
        #<start> l action pour les autres editions de ce texte <end>
        #1
        #<start> l action restreinte . petit air guerrier ce me <end>
        #<start> voir l action restreinte . petit air guerrier ce <end>
        #2
        #<start> hormis l y taire que je sente du foyer <end>
        #<start> va hormis l y taire que je sente du <end>

        self.num_examples=len(targ_sequences)
        input_tensor, inp_tokenizer = self.tokenize(targ_sequences)
        target_tensor, targ_tokenizer = self.tokenize(inp_sequences)
        return input_tensor, target_tensor, inp_tokenizer, targ_tokenizer 

    def call(self, BUFFER_SIZE, BATCH_SIZE):
        
        file_path = FILE_PATH        

        input_tensor, target_tensor, self.inp_tokenizer, self.targ_tokenizer = self.load_dataset(file_path)
        input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

        train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        val_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))
        val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

        return train_dataset, val_dataset, self.inp_tokenizer, self.targ_tokenizer

dataset_creator = Seq2seqTextGenDataset()
train_dataset, val_dataset, inp_tokenizer, targ_tokenizer = dataset_creator.call(BUFFER_SIZE, BATCH_SIZE)

example_input_batch, example_target_batch = next(iter(train_dataset))
example_input_batch.shape, example_target_batch.shape

vocab_inp_size = len(inp_tokenizer.word_index)+1
vocab_targ_size = len(targ_tokenizer.word_index)+1
max_length_input = example_input_batch.shape[1]
max_length_output = example_target_batch.shape[1]

print("max_length_input, max_length_target, vocab_inp_size, vocab_targ_size")
print(str(max_length_input)+", "+str(max_length_output)+", "+str(vocab_inp_size)+", "+str(vocab_targ_size))

steps_per_epoch = dataset_creator.num_examples//BATCH_SIZE

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

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

sample_hidden = encoder.initialize_hidden_state()# sample input
sample_output, sample_h, sample_c = encoder(example_input_batch, sample_hidden)# sample output
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder h vecotr shape: (batch size, units) {}'.format(sample_h.shape))
print ('Encoder c vector shape: (batch size, units) {}'.format(sample_c.shape))

## Decoder stack

# Custom sampler

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
                                                              None, self.batch_sz*[max_length_input], self.attention_type)

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
    # memory: encoder hidden states of shape (batch_size, max_length_input, enc_units)
    # memory_sequence_length: 1d array of shape (batch_size) with every element set to max_length_input (for masking purpose)

    if(attention_type=='bahdanau'):
      return tfa.seq2seq.BahdanauAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)
    else:
      return tfa.seq2seq.LuongAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)

  def build_initial_state(self, batch_sz, encoder_state, Dtype):
    decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_sz, dtype=Dtype)
    decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
    return decoder_initial_state


  def call(self, inputs, initial_state):
    x = self.embedding(inputs)
    outputs, _, _ = self.decoder(x, initial_state=initial_state, sequence_length=self.batch_sz*[max_length_output-1])
    return outputs

# Test decoder stack

decoder = Decoder(vocab_targ_size, embedding_dim, units, BATCH_SIZE, 'luong')
sample_x = tf.random.uniform((BATCH_SIZE, max_length_output))
decoder.attention_mechanism.setup_memory(sample_output)
initial_state = decoder.build_initial_state(BATCH_SIZE, [sample_h, sample_c], tf.float32)

sample_decoder_outputs = decoder(sample_x, initial_state)

print("Decoder Outputs Shape: ", sample_decoder_outputs.rnn_output.shape)

# Defining optimizer and loss function

optimizer = tf.keras.optimizers.Adam()

def loss_function(real, pred):
  # real shape = (BATCH_SIZE, max_length)
  # pred shape = (BATCH_SIZE, max_length, tar_vocab_size )
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
manager = tf.train.CheckpointManager(
    checkpoint, directory=checkpoint_dir, max_to_keep=3)

@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_h, enc_c = encoder(inp, enc_hidden)

    dec_input = targ[ : , :-1 ] # Ignore <end> token
    real = targ[ : , 1: ]       # ignore <start> token

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



############ FOR GENERATION

def evaluate_sentence(text):

  if text[-6:] == ' <end>':
      text=text[:-6]
    
  text = dataset_creator.preprocess_text(text)

  text = '<start> ' + text + ' <end>'
  
  inputs = [inp_tokenizer.word_index[i] for i in text.split(' ')]

  maxlen=len(inputs)

  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                          maxlen=maxlen,
                                                          padding='post')

  inputs = tf.convert_to_tensor(inputs)
  inference_batch_size = inputs.shape[0]
  result = ''

  enc_start_state = [tf.zeros((inference_batch_size, units)), tf.zeros((inference_batch_size,units))]
  enc_out, enc_h, enc_c = encoder(inputs, enc_start_state)

  dec_h = enc_h
  dec_c = enc_c

  start_tokens = tf.fill([inference_batch_size], targ_tokenizer.word_index['<start>'])
  end_token = targ_tokenizer.word_index['<end>']

  greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

  # Instantiate BasicDecoder object
  decoder_instance = tfa.seq2seq.BasicDecoder(cell=decoder.rnn_cell, sampler=greedy_sampler, output_layer=decoder.fc)
  # Setup Memory in decoder stack
  decoder.attention_mechanism.setup_memory(enc_out)

  # set decoder_initial_state
  decoder_initial_state = decoder.build_initial_state(inference_batch_size, [enc_h, enc_c], tf.float32)

  ### Since the BasicDecoder wraps around Decoder's rnn cell only, you have to ensure that the inputs to BasicDecoder 
  ### decoding step is output of embedding layer. tfa.seq2seq.GreedyEmbeddingSampler() takes care of this. 
  ### You only need to get the weights of embedding layer, which can be done by decoder.embedding.variables[0] and pass this callabble to BasicDecoder's call() function

  decoder_embedding_matrix = decoder.embedding.variables[0]

  outputs, _, _ = decoder_instance(decoder_embedding_matrix, start_tokens = start_tokens, end_token= end_token, initial_state=decoder_initial_state)
  return outputs.sample_id.numpy()

def generate_next_word(sentence):
  result = evaluate_sentence(sentence)
  result = targ_tokenizer.sequences_to_texts(result)
  print('Input: %s' % (sentence))
  print('Predicted output: {}'.format(result))
  return result

words = u'Agir, sans ceci et pour qui n’en fait commencer l’exercice à fumer, \
signifia, visiteur, je te comprends, philosophiquement, produire sur beaucoup un mouvement \
qui te rendre en retour l’émoi que tu en fus le principe, donc existes : dont aucun ne se croit, \
au préalable, sûr. Cette pratique entend deux façons ; ou, par une volonté, à l’insu, qui dure une vie, \
jusqu’à l’éclat multiple — penser, cela : sinon, les déversoirs à portée maintenant dans une prévoyance, \
supérieure à toute édilité, journaux et leur tourbillon, y déterminer une force vers un sens, \
quelconque de cent contrariée, avec l’immunité du résultat nul. \
Au gré, selon la disposition, plénitude, vacuité.'

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(manager.latest_checkpoint)
 
words = generate_next_word(words)    

## print to file

i = 0
while os.path.exists("mallarme-like-Seq2Seq-V5-%s.txt" % i):
    i += 1
    
with open("mallarme-like-Seq2Seq-V5-%s.txt" % i,'w+') as f:
  f.write("embedding dim: " + str(embedding_dim)+"\n")
  f.write("batch size: " + str(BATCH_SIZE)+"\n")
  f.write("rnn units: " + str(units)+"\n")
  f.write("sequence length: " + str(max_words)+"\n")
  f.write("epochs: " + str(EPOCHS)+"\n\n"+'_'*80+"\n")
  f.write(words[0] + '\n\n' + '_'*80)
