#################################

## TRAINING

#################################


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
sequence_length = 500
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
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Define the checkpoint ---
filepath = "improved-weights-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

EPOCHS = 50
BATCH_SIZE = 128

# Fit the model to data
history = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_list)

#Summarize the loss history---
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

