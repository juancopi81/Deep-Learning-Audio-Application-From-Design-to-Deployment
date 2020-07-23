import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

DATA_PATH = "data.json"
LEARNING_RATE = 0.0001
EPOCHS = 40
BATCH_SIZE = 32
SAVED_MODEL_PATH = "model.h5"
NUM_KEYWORDS = 10


def load_dataset(data_path):

	with open(data_path, "r") as fp:
		data = json.load(fp)

	# Extract inputs and targets
	X = np.array(data["MFCCs"])
	y = np.array(data["labels"])

	return X, y


def get_data_splits(data_path, test_size=0.1, validation_size=0.1):

	# Load the dataset
	X, y = load_dataset(data_path)

	# Create the train, validation and test splits
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
	X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

	# Convert inputs from 2D to 3D arrays
	# (# segments, # MFFCs 13, we want to add new dim as 1)
	X_train = X_train[..., np.newaxis]
	X_validation = X_validation[..., np.newaxis]
	X_test = X_test[..., np.newaxis]

	return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape, learning_rate, error="sparse_categorical_crossentropy"):

	# build the network
	model = tf.keras.Sequential()

	# conv layer 1
	model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))

	# conv layer 2
	model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))

	# conv layer 3
	model.add(tf.keras.layers.Conv2D(32, (2, 2), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))

	# flatten the output feed it into a dense layer
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(64, activation="relu"))
	model.add(tf.keras.layers.Dropout(0.3))

	# Softmas classifier
	model.add(tf.keras.layers.Dense(NUM_KEYWORDS, activation="softmax")) # [0.1, 0.7, 0.2]

	# compile the model
	optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)
	model.compile(optimizer=optimiser, loss=error, metrics=["accuracy"])

	# print model summary
	model.summary()

	return model


def main():

	# Load train/validation/test data splits
	X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits(DATA_PATH)  

	# Build the CNN model
	input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) # (# segments, # Coefficients 13, 1 -> Depth / Channel)
	model = build_model(input_shape, LEARNING_RATE)

	# Train the model
	model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_validation, y_validation))

	# Evaluate the model
	test_error, test_accuracy = model.evaluate(X_test, y_test)
	print(f"Test error = {test_error}, test accuracy = {test_accuracy}")

	# Save the model
	model.save(SAVED_MODEL_PATH)


if __name__=="__main__":
	main()
