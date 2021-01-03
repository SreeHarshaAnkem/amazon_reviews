import os
import time
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
logging.basicConfig(filename="../logs/train.log", level = logging.INFO)

def load_datasets(file_path, num_samples):
	df = pd.read_csv(file_path, usecols=[6,9], 
			nrows=num_samples)
	text = df.iloc[:,1].values.tolist()
	text = np.array([str(t).encode("ascii", "replace") 
				for t in text], dtype="object")
	labels = df.iloc[:, 0].values.tolist()
	labels = [1 if i>=4 else 0 if i==3 else -1 for i in labels]
	labels = np.array(pd.get_dummies(labels, dtype=int)) 
	return text, labels

def get_model():
	hub_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1",
				   input_shape =[], output_shape=50, name="input",
				   dtype=tf.string)
	model = tf.keras.models.Sequential()
	model.add(hub_layer)
	model.add(tf.keras.layers.Dense(units=10,
					activation="relu"))
	model.add(tf.keras.layers.Dense(units=3, 
					activation="softmax", 
					name="output"))

	model.compile(loss = "categorical_crossentropy",
		      optimizer="adam", metrics=["accuracy"])
	logging.info(model.summary())
	return model

def train(batch_size, epochs, model_dir):
	if not os.path.exists(model_dir):
		os.mkdir(model_dir)

	model_dir = os.path.join(model_dir, str(time.time()))
	x_train, y_train = load_datasets("../data/train.csv", num_samples=100000)
	x_test, y_test = load_datasets("../data/test.csv", num_samples=10000)
	logging.info("loaded datasets")
	model = get_model()
	logging.info("creating architecture")
	model.fit(x_train, y_train, validation_data=(x_test, y_test),
		 batch_size=batch_size, epochs=epochs,
		 callbacks = [tf.keras.callbacks.ModelCheckpoint(model_dir, 
 								save_best_only=True,
								save_weights_only=False,
								mode="auto",
								monitor="val_loss",
								verbose=1)])
	return  model

def export_model(model, model_save_path):
	tf.saved_model.save(model, model_save_path)
	



if __name__ == "__main__":
	model = train(batch_size=32, epochs=5, model_dir="../model_checkpoint/")
	logging.info("Finished training")
	export_model(model, model_save_path = "../saved_model/")
	logging.info("Saved moved assets")

