import tensorflow as tf
import tensorflow_hub as hub

def load_datasets():
	
	return x, y


def get_model():
	hub_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1",
				   input_shape =[], output_shape=50, name="input")

	model = tf.keras.model.Sequential()
	model.add(hub_layer)
	model.add(tf.keras.layers.Dense(units=10, activation="relu"))
	model.add(tf.keras.layers.Dense(units=3, activation="softmax"))

	model.compile(loss = "categorical_crossentropy",optimizer="adam", metrics=["accuracy"])
	
	print(model.summary())

	return model

def train(model, x,y,batch_size, epochs, model_dir):
	model.fit(x, y, batch_size=batch_size, epochs=epochs,
		 callbacks = [tf.keras.callbacks.ModelCheckpoint(model_dir, 
 								save_best_only=True,
								save_weights_only=False,
								mode="auto")
def export_model(model, model_save_path):
	tf.saved_model.save(model, model_save_path)
	



if __name__ == "__main__":
	load_datasets()
