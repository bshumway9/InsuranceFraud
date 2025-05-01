#!/usr/bin/env python3

#num layers, width layers, activation functions
#layer weight initializers
#keras layer activation functions 
#activation swish

import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

label = "FraudFound_P"
input_filename = "preprocessed-train.csv"
model_filename = "model.keras"
train_ratio = 0.80
learning_curve_filename = "learning-curve.png"
#
# Load the training dataframe, separate into X/y
#
dataframe = pd.read_csv(input_filename, index_col=0)
X = dataframe
y = dataframe[label]

# print(dataframe)
# print(X)
# print(y)

#
# Prepare a tensorflow dataset from the dataframe
#
dataset = tf.data.Dataset.from_tensor_slices((X, y))
# print(dataset)
# print(list(dataset.as_numpy_iterator()))
# print(dataset.element_spec)


#
# Find the shape of the inputs and outputs.
# Necessary for the model to have correctly sized input and output layers
#
#
# This is happening *before* batching, so
# the shape does not yet include the batch size
#
for features, labels in dataset.take(1):
    input_shape = features.shape
    output_shape = labels.shape
# print(input_shape)
# print(output_shape)


#
# Split the dataset into train and validation sets.
#
dataset_size       = dataset.cardinality().numpy()
train_size         = int(train_ratio * dataset_size)
validate_size      = dataset_size - train_size
train_dataset      = dataset.take(train_size)
validation_dataset = dataset.skip(train_size)

#
# Cause the datasets to shuffle, internally
#
train_dataset      = train_dataset.shuffle(buffer_size=train_size)
validation_dataset = validation_dataset.shuffle(buffer_size=validate_size)

#
# Cause the datasets to batch.
# Efficiency benefits.
# Training differences.
#
BATCH_SIZE = 64
train_dataset      = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


#activation initilaizer combinations:
#linear glorot
#sigmoid glorot
#hyperbolic tanh glorot
#relu he
#leaky relu he
#exponential linear unit he
#scaled exponential linear unit leCun
#guassian error linear unit he
#swish he
#mish he

#
# Build the model
#

activation = "swish"
#
initializer = "he_normal"
units = 15
layer_count = 3
#units 20 layer count 5 for submission 3
#units 15 layer count 3 for submission 2
#units 10 layer count 1 for submission 1
activation, initializer = "relu", "glorot_uniform"
#relu, glorot_normal for submission 1
#try 2 used swish and he_normal
#relu he_normal for submission 2
#num 3 used swish and he_normal
# activation, initializer = "relu", "he_normal"
tf.random.set_seed(70982173)
#submission 3 used seed 70982173
#sub 1 and 2 used 42
model = keras.Sequential()
model.add(keras.layers.Input(shape=input_shape))
for i in range(layer_count):
    # model.add(keras.layers.BatchNormalization())
    #after submission 2 added batch normalization before each dense layer
    model.add(keras.layers.Dense(units, activation=activation, kernel_initializer=initializer))
#use dropout layers to help with overfitting
#use stochastic gradient descent also
# model.add(keras.layers.Dense(units, activation="relu"))
#submission 1 sigmoid
# model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(1, activation="sigmoid"))
#try 3 used swish for final layer but was really bad
#try 2 used sigmoid
#submission 1 last layer sigmoid
# model.add(keras.layers.Dense(1, activation="linear"))
#print(model.summary())
#print(model.layers[1].get_weights())

#
# Compile the model
#
loss = "binary_crossentropy"
#submission 2 and 3 binary cross entropy
# loss = "mean_squared_error"
#submission 1 mean squared error
# loss = "mean_absolute_error"
model.compile(loss=loss,
              optimizer=keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True),
              metrics=["AUC", "accuracy", "r2_score"])
#submission 3 adagrad optimizer and AUC metric and learning rate 0.1
#submission 2 metric AUC learning rate 0.1
#try 3 used 0.2
#submission 1 metric r2score learning rate 0.1
            #   optimizer=keras.optimizers.SGD(learning_rate=0.1),
#check for metric R2Score
#was AUC


#
# Update the learning rate dynamically
#
def scheduler(epoch, learning_rate):
    r = learning_rate
    if epoch >= 10:
        r = learning_rate * float(tf.exp(-0.005))
    return r
learning_rate_callback = keras.callbacks.LearningRateScheduler(scheduler)

#
# Stop training if validation loss does not improve
#
early_stop_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

#
# Train for up to epoch_count epochs
#
epoch_count = 15
history = model.fit(x=train_dataset,
                    epochs=epoch_count,
                    validation_data=validation_dataset,
                    callbacks=[learning_rate_callback, early_stop_callback])
epochs = len(history.epoch)
# print(history)


#
# Display the learning curves
#
line_style = ["r--", "r--*", "r--+", "b-", "b-*", "b-+"]
line_style = ["r--*", "r--+", "b-*", "b-+"]
pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[0, epochs-1], ylim=[0, 1], grid=True, xlabel="Epoch",
    style=line_style)
# plt.show()
plt.savefig(learning_curve_filename)
plt.clf()


#
# Save the model
#
model.save(model_filename)
