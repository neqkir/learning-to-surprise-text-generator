## neqkir

# checks what percentage of the output text locates in the initial vocabulary
# it basically tests the proportion of neologisms

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

import numpy as np
import os
import time



##### Read and word-tokenize the input train text

input_text = open("mallarme.txt", 'rb').read().decode(encoding='utf-8')

t_tokenizer = Tokenizer()
t_tokenizer.fit_on_texts([input_text])#Builds the word index
t_word_index=t_tokenizer.word_index 

t_vocab_size = len(t_word_index) + 1

print ("Training vocabulary: "+str(t_vocab_size)+" words.")



##### Read and word-tokenize the generated text

generated_text = open("mallarme-like.txt", 'rb').read().decode(encoding='utf-8')

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

with open('mallarme-like-w-metrics.txt','a') as f:
  f.write(output_str + '\n\n' + '_'*80)
  f.write(generated_text + '\n\n' + '_'*80)
