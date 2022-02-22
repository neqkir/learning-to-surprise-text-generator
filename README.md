# learning-to-surprise-text-generator
Learning to Surprise: A Composer-Audience Architectur

We start from a survey of creative generative AIs 

Giorgio Franceschelli, Mirco Musolesi:
Creativity and Machine Learning: A Survey. CoRR abs/2104.02726 (2021)

https://arxiv.org/abs/2104.02726 

![image](https://user-images.githubusercontent.com/89974426/154297377-e8357a34-9a5b-45d1-95b8-661b71a3d4cd.png)

We develop a method for generating creative text according to Boden's definition of creativity. 

*“Boden identifies three forms of creativity: combinatorial, exploratory and transformational creativity. Combinatorial creativity is about making unfamiliar combinations of familiar ideas; the definition of the other two, instead, is based on conceptual spaces, i.e., structured styles of thought (any disciplined way of thinking that is familiar to a certain culture or peer group): exploratory creativity involves the exploration of the structured conceptual space, while transformational creativity involves changing the conceptual space in a way that new thoughts, previously inconceivable, become possible.”*

Deep Creator must have a combinatorial and exploratory creativity. Transformational is of secondary importance because we want Deep Creator to produce according to some author or in the style of. We can therefore afford to stay in the same space. Deep Creator should also show creative within its pasture, featuring all aspects of creativity, in particularly the much sought after surprise.  

While GANs, LSTMs, transformers produce new and valuable artifacts, they lack sparkles of surprise. An audience-composer architecture seems the framework of choice to instill surprise, with an exploratory behavior, and we will therefore explore this direction.

### RNN

We start with a simple char-RNN generator. One GRU layer with 2048 neuronal units. We play with parameters, we play with states : first we force states to be reset to zero at the end of each epoch only. By default, Keras reset states for each batch. This is elegantly implemented in RRN/V3 with custom callbacks.

```
gru_layer = model.layers[1]
class CustomCallback(tf.keras.callbacks.Callback):
   def __init__(self, gru_layer):
        self.gru_layer = gru_layer
   def on_epoch_end(self, epoch, logs=None):
        self.gru_layer.reset_states()

model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, \
   callbacks = [EarlyS, CustomCallback(gru_layer, BATCH_SIZE, rnn_units)], verbose=1)

```

Another possible implementation was (see commit history)

```
for i in range(EPOCHS):
  model.fit(train_dataset, validation_data=validation_dataset, epochs=1, callbacks = [EarlyS], verbose=1)
  model.reset_states()
```

less elegant and less efficient. In RNN/V4 we initialize states to random noise and try different kind of noises

```
gru_layer = model.layers[1]

class CustomCallback(tf.keras.callbacks.Callback):
   def __init__(self, gru_layer, batch_size, dims):
        self.gru_layer = gru_layer
        self.batch_size = batch_size
        self.dims = dims
   def on_epoch_end(self, epoch, logs=None):
        self.gru_layer.reset_states(tf.random.normal((self.batch_size, self.dims), stddev=0.5, mean=0, seed=42))        

model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, \
     callbacks = [EarlyS, CustomCallback(gru_layer, BATCH_SIZE, rnn_units)], verbose=1)
```

