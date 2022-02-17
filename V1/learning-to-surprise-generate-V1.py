###################################

### GENERATOR

###################################

# Import libraries--------
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# Read the file
file = open("mallarme.txt")
text = file.read()
text = text.lower()
#print(text)

# create the mapping of character to integer
number_of_chars = sorted(list(set(text)))
char_to_int = dict((c, i) for i, c in enumerate(number_of_chars))
int_to_char = dict((i, c) for i, c in enumerate(number_of_chars) )
#print(chars)
#print(char_to_int)
# pass
total_chars = len(text)
total_vocab = len(number_of_chars)
#print("Total Characters: ", total_chars)
#print("Total Vocabulary: ", total_vocab)

# Prepare the dataset pairs encoded as integers
sequence_length = 150
dataX = []
dataY = []
for i in range(0, total_chars - sequence_length, 1):
  sequence_in = text[i : i + sequence_length]
  sequence_out = text[i + sequence_length]
  dataX.append([char_to_int[char] for char in sequence_in])
  dataY.append(char_to_int[sequence_out])
number_of_patterns = len(dataX)
#print("Total Patterns: ", number_of_patterns)

# Step-1: transform the list input sequence
X = numpy.reshape(dataX, (number_of_patterns, sequence_length, 1))
#Step-2: rescale the input
X = X / float(total_vocab)
#Step-3: convert the output patterns into one-hot-encodding
y = np_utils.to_categorical(dataY)

# Define the model---
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
# Load the network ---
filename = "improved-weights-50-1.8569.hdf5" 
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# Pick a random seed
start = numpy.random.randint(0,  len(dataX) - 1)
pattern = dataX[start]
print ("Seed: ")
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
# Generate Chatacters
text = ""
for i in range(1000):
  x = numpy.reshape(pattern, (1, len(pattern), 1))
  x = x/float(total_vocab)
  prediction = model.predict(x, verbose=0)
  index = numpy.argmax(prediction)
  result = int_to_char[index]
  text += result
  print(result,end="")
  seq_in = [int_to_char[value] for value in pattern]
  pattern.append(index)
  pattern = pattern[1: len(pattern)]

with open("text.txt", "w", encoding="utf8") as file:
  file.write(text)

print ("\nDone")
