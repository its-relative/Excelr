#!/usr/bin/env python
# coding: utf-8

# ### Import the data set

# In[1]:


import pandas as pd
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
train_df.head()


# In[3]:


train_df.head(50)


# ##### # How many samples of each class?

# In[2]:


train_df.target.value_counts()


# In[3]:


train_df_shuffled = train_df.sample(frac=1, random_state=42) # shuffle with random_state=42 for reproducibility
train_df_shuffled.head()


# ### Let's visualize some random training examples

# In[4]:


import random
random_index = random.randint(0, len(train_df)-5) # create random indexes not higher than the total number of samples
for row in train_df_shuffled[["text", "target"]][random_index:random_index+5].itertuples():
  _, text, target = row
  print(f"Target: {target}", "(real disaster)" if target > 0 else "(not real disaster)")
  print(f"Text:\n{text}\n")
  print("---\n")


# In[5]:


from sklearn.model_selection import train_test_split

# Use train_test_split to split training data into training and validation sets
train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_df_shuffled["text"].to_numpy(),
                                                                            train_df_shuffled["target"].to_numpy(),
                                                                            test_size=0.1,
                                                                            random_state=42) # random state for reproducibility


# In[6]:


# Check the lengths
len(train_sentences), len(train_labels), len(val_sentences), len(val_labels)


# In[7]:


import tensorflow as tf
from tensorflow.keras.layers import TextVectorization # after TensorFlow 2.6

# Before TensorFlow 2.6
# from tensorflow.keras.layers.experimental.preprocessing import TextVectorization 
# Note: in TensorFlow 2.6+, you no longer need "layers.experimental.preprocessing"
# you can use: "tf.keras.layers.TextVectorization"

# Use the default TextVectorization variables
text_vect = TextVectorization(max_tokens=None, # how many words in the vocabulary (all of the different words in your text)
                                    standardize="lower_and_strip_punctuation", # how to process text
                                    split="whitespace", # how to split tokens
                                    ngrams=None, # create groups of n-words?
                                    output_mode="int", # how to map tokens to numbers
                                    output_sequence_length=None) # how long should the output sequence of tokens be?


# In[8]:


# Fit the text vectorizer to the training text
text_vect.adapt(train_sentences)


# In[9]:


# Create sample sentence and tokenize it
sample_sentence = "There is flood in my city"
text_vect([sample_sentence])


# In[10]:


# Create sample sentence and tokenize it
sample_sentence = "There is flood in my city and we are looking for help"
text_vect([sample_sentence])


# In[ ]:





# In[11]:


# Find average number of tokens (words) in training Tweets
round(sum([len(i.split()) for i in train_sentences])/len(train_sentences))


# In[12]:


# Setup text vectorization with custom variables
max_vocab_length = 10000 # max number of words to have in our vocabulary (most common words)
max_length = 15 # max length our sequences will be (e.g. how many words from a Tweet does our model see?)

text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                    output_mode="int",
                                    output_sequence_length=max_length)


# In[13]:


# Fit the text vectorizer to the training text
text_vectorizer.adapt(train_sentences)


# In[14]:


# Create sample sentence and tokenize it
sample_sentence = "There is flood in my city"
text_vectorizer([sample_sentence])
# to match the output sequence(so that we can feed those same length vectorized as input)
# length it has to generate the remaining 0's.


# In[15]:


# Create sample sentence and tokenize it
sample_sentence = "There is flood in my city and we are looking for help"
text_vectorizer([sample_sentence])


# ### DRAWBACKS of Textvectorization: 
# ###           1. creats very huge matrix
# ###           2. results in sparse matrix representation
# ###           3. provides static vector representation

# In[ ]:





# ## Word Embedding

# In[16]:


tf.random.set_seed(42)
from tensorflow.keras import layers

embedding = layers.Embedding(input_dim=max_vocab_length, # set input shape
                             output_dim=128, # set size of embedding vector
                             embeddings_initializer="uniform", # default, intialize randomly
                             input_length=max_length, # how long is each input
                             name="embedding_1") 



# In[17]:


sample_sentence = "There is flood in my city"
sample_embed = embedding(text_vectorizer([sample_sentence]))
sample_embed


# In[18]:


# Check out a single token's embedding
sample_embed[0][0]


# ### LSTM

# In[68]:


# Create LSTM model
inputs = layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = embedding(x)
x = layers.LSTM(64)(x) # return vector for whole sequence
x = layers.Dense(64, activation="relu")(x) # optional dense layer on top of output of LSTM cell
outputs = layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs, name="model_2_LSTM")


# In[69]:


# Compile model
model.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])


# In[70]:


# Fit model
model_history = model.fit(train_sentences,
                              train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels))


# In[71]:


# Make predictions on the validation dataset
model_pred_probs = model.predict(val_sentences)
model_pred_probs.shape, model_pred_probs[:10] # view the first 10


# In[72]:


### We can turn these prediction probabilities into prediction classes by rounding to the nearest integer 
### (by default, prediction probabilities under 0.5 will go to 0 and those over 0.5 will go to 1).

# Round out predictions and reduce to 1-dimensional array
model_1_preds = tf.squeeze(tf.round(model_pred_probs))
model_1_preds[:10]


# In[73]:


# Function to evaluate: accuracy, precision, recall, f1-score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results(y_true, y_pred):
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
    return model_results


# In[74]:


model_1_results = calculate_results(val_labels, model_1_preds)
model_1_results


# In[ ]:





# ### Model 2: GRU

# * Another popular and effective RNN component is the GRU or gated recurrent unit.
# 
# * The GRU cell has similar features to an LSTM cell but has less parameters.

# In[75]:


inputs = layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = embedding(x)
x = layers.GRU(64)(x) # return vector for whole sequence
x = layers.Dense(64, activation="relu")(x) # optional dense layer on top of output of LSTM cell
outputs = layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs, name="model_2_LSTM")


# In[76]:


# Compile GRU model
model.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])


# In[77]:


model_history = model.fit(train_sentences,
                              train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels))


# In[78]:


# Make predictions on the validation dataset
model_pred_probs = model.predict(val_sentences)
model_pred_probs.shape, model_pred_probs[:10]


# In[79]:


model_2_preds = tf.squeeze(tf.round(model_pred_probs))
model_2_preds[:10]


# In[80]:


model_2_results = calculate_results(val_labels, model_2_preds)
model_2_results


# In[ ]:





# In[ ]:





# ### Model 3: Bidirectonal RNN model

# * A standard RNN will process a sequence from left to right, where as a bidirectional RNN will process the sequence from left to right and then again from right to left.
#  * In practice, many sequence models often see and improvement in performance when using bidirectional RNN's.
# 
# * However, this improvement in performance often comes at the cost of longer training times and increased model parameters (since the model goes left to right and right to left, the number of trainable parameters doubles).

# In[31]:


# Set random seed and create embedding layer (new embedding layer for each model)
tf.random.set_seed(42)
from tensorflow.keras import layers
model_4_embedding = layers.Embedding(input_dim=max_vocab_length,
                                     output_dim=128,
                                     embeddings_initializer="uniform",
                                     input_length=max_length,
                                     name="embedding_4")

# Build a Bidirectional RNN in TensorFlow
inputs = layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = model_4_embedding(x)
# x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x) # stacking RNN layers requires return_sequences=True
x = layers.Bidirectional(layers.LSTM(64))(x) # bidirectional goes both ways so has double the parameters of a regular LSTM layer
outputs = layers.Dense(1, activation="sigmoid")(x)
model_4 = tf.keras.Model(inputs, outputs, name="model_4_Bidirectional")


# In[32]:


# Compile
model_4.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])


# In[33]:


# Fit the model (takes longer because of the bidirectional layers)
model_4_history = model_4.fit(train_sentences,
                              train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels))


# In[34]:


# Make predictions with bidirectional RNN on the validation data
model_4_pred_probs = model_4.predict(val_sentences)
model_4_pred_probs[:10]


# In[35]:


# Convert prediction probabilities to labels
model_4_preds = tf.squeeze(tf.round(model_4_pred_probs))
model_4_preds[:10]


# In[82]:


model_4_results = calculate_results(val_labels, model_4_preds)
model_4_results


# In[ ]:





# ### Conv1D

# In[37]:


# Set random seed and create embedding layer (new embedding layer for each model)
tf.random.set_seed(42)
from tensorflow.keras import layers
model_5_embedding = layers.Embedding(input_dim=max_vocab_length,
                                     output_dim=128,
                                     embeddings_initializer="uniform",
                                     input_length=max_length,
                                     name="embedding_5")


# In[40]:


# Create 1-dimensional convolutional layer to model sequences
from tensorflow.keras import layers

inputs = layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = model_5_embedding(x)
x = layers.Conv1D(filters=32, kernel_size=5, activation="relu")(x)
x = layers.GlobalMaxPool1D()(x)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model_5 = tf.keras.Model(inputs, outputs, name="model_5_Conv1D")


# In[43]:


# Compile Conv1D model
model_5.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])


# In[44]:


# Fit the model
model_5_history = model_5.fit(train_sentences,
                              train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels))


# In[46]:


model_5.summary()


# In[49]:


# Make predictions with model_5
model_5_pred_probs = model_5.predict(val_sentences)
model_5_pred_probs[:10]


# In[50]:


# Convert model_5 prediction probabilities to labels
model_5_preds = tf.squeeze(tf.round(model_5_pred_probs))
model_5_preds[:10]


# In[83]:


model_5_results = calculate_results(val_labels, model_5_preds)
model_5_results


# In[ ]:





# ### Using Pretrained Embeddings (transfer learning for NLP)
# 
# * common practice is to leverage pretrained embeddings through transfer learning. This is one of the main benefits of using deep models: being able to take what one (often larger) model has learned (often on a large amount of data) and adjust it for our own use case.
# * Universal Sentence Encoder:
#     * Universal Sentence Encoder input is of variable length
#     * Universal Sentence Encoder outputs a 512 dimensional vector for each sentence.

# In[56]:


#!pip install tensorflow_hub


# In[57]:


import tensorflow_hub as hub

sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                        input_shape=[], # shape of inputs is variable
                                        dtype=tf.string, # data type of inputs coming to the USE layer
                                        trainable=False, # keep the pretrained weights (we'll create a feature extractor)
                                        name="USE")


# In[60]:


# Create model using the Sequential API
model_6 = tf.keras.Sequential([
                              sentence_encoder_layer, # take in sentences and then encode them into an embedding
                              layers.Dense(64, activation="relu"),
                              layers.Dense(32, activation="relu"),
                              layers.Dense(1, activation="sigmoid")], name="model_6_USE")


# In[61]:


# Compile model
model_6.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])


# In[62]:


# Train a classifier on top of pretrained embeddings
model_6_history = model_6.fit(train_sentences,
                              train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels))


# In[63]:


# Make predictions with USE TF Hub model
model_6_pred_probs = model_6.predict(val_sentences)
model_6_pred_probs[:10]


# In[64]:


# Convert prediction probabilities to labels
model_6_preds = tf.squeeze(tf.round(model_6_pred_probs))
model_6_preds[:10]


# In[67]:


model_6_results = calculate_results(val_labels, model_6_preds)
model_6_results


# In[ ]:





# In[ ]:





# ## Comparing the performance of each of our models

# In[84]:


# Combine model results into a DataFrame
all_model_results = pd.DataFrame({"LSTM_Model": model_1_results,
                                  "GRU_Model": model_2_results,
                                  "Bidirectional RNN_Model": model_4_results,
                                  "conv1d": model_5_results,
                                  "USE_Encoder_Model": model_6_results})

all_model_results = all_model_results.transpose()
all_model_results


# In[86]:


# Reduce the accuracy to same scale as other metrics
all_model_results["accuracy"] = all_model_results["accuracy"]/100


# In[87]:


# Plot and compare all of the model results
all_model_results.plot(kind="bar", figsize=(10, 7)).legend(bbox_to_anchor=(1.0, 1.0));


# In[ ]:





# ### Saving and loading a trained model
# 
# * There are two main ways of saving a model in TensorFlow:
# 
#     * The HDF5 format.
#     * The SavedModel format (default).

# In[ ]:


# Save TF Hub Sentence Encoder model to HDF5 format
model_6.save("model_6.h5")


# In[ ]:


loaded_model_6 = tf.keras.models.load_model("model_6.h5")


# In[ ]:


# How does our loaded model perform?
loaded_model_6.evaluate(val_sentences, val_labels)


# In[ ]:





# In[ ]:


# Save TF Hub Sentence Encoder model to SavedModel format (default)
model_6.save("model_6_SavedModel_format")


# In[ ]:


# Load TF Hub Sentence Encoder SavedModel
loaded_model_6_SavedModel = tf.keras.models.load_model("model_6_SavedModel_format")


# In[ ]:


# Evaluate loaded SavedModel format
loaded_model_6_SavedModel.evaluate(val_sentences, val_labels)


# In[ ]:





# In[ ]:




