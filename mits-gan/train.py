import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import save_model

import config

# Disable GPU if needed (comment this out if you want to use GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Reduce TensorFlow logging verbosity
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Configure threading
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Load data
XAuthenticate = list(np.load(config.np_casia_two_au_path))
yAuthenticate = list(np.zeros(len(XAuthenticate), dtype=np.uint8))

XForged = list(np.load(config.np_casia_two_forged_path))
yForged = list(np.ones(len(XForged), dtype=np.uint8))

X = np.array(XAuthenticate + XForged)
y = np.array(yAuthenticate + yForged, dtype=np.int8)

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, shuffle=True
)

# Visualize class distribution
plt.hist(y, bins=5)
plt.ylabel("Number of images")
plt.title("CASIA II - Authenticate OR Fake Image Dataset")
plt.savefig("class_distribution.png")  # Save the plot
plt.close()

# Image dimensions
img_height = 256
img_width = 384

# Create model
base_model = VGG16(
    weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3)
)

top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation="relu"))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation="sigmoid"))

new_model = Sequential()
for layer in base_model.layers:
    new_model.add(layer)
new_model.add(top_model)

# Freeze early layers
for layer in new_model.layers[:15]:
    layer.trainable = False

print("Model loaded and layers frozen.")
print(new_model.summary())

# Compile model
new_model.compile(
    loss="binary_crossentropy",
    optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.9),
    metrics=["accuracy"],
)

# Create directory for saving model
model_dir = os.path.join(os.getcwd(), "saved_models")
os.makedirs(model_dir, exist_ok=True)

# Checkpoint to save model during training
checkpoint_path = os.path.join(model_dir, "model_checkpoint")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    save_best_only=True,
    monitor="accuracy",
    verbose=1,
)

# Train model
print("Starting training...")
history = new_model.fit(
    x_train,
    y_train,
    epochs=2,  # Temporarily set to 2 epochs for testing
    batch_size=10,
    verbose=1,  # Show progress per epoch
    callbacks=[checkpoint_callback],
)
print("Finished training!")

# Evaluate model
loss, acc = new_model.evaluate(x_test, y_test, verbose=1)
print(f"Test Accuracy: {acc:.4f}")

# Predict
y_pred = (new_model.predict(x_test) > 0.5).astype("int32")

# Save model in multiple formats for redundancy
try:
    # Method 1: SavedModel format (preferred)
    save_path_tf = os.path.join(model_dir, "casia2_model_savedmodel")
    print(f"Saving model to: {save_path_tf}")
    new_model.save(save_path_tf, save_format="tf")
    print("Model saved in SavedModel format!")

    # Method 2: H5 format
    save_path_h5 = os.path.join(model_dir, "casia2_model.h5")
    print(f"Saving model to: {save_path_h5}")
    new_model.save(save_path_h5, save_format="h5")
    print("Model saved in H5 format!")

    # Method 3: Save weights separately
    weights_path = os.path.join(model_dir, "casia2_model_weights.h5")
    print(f"Saving model weights to: {weights_path}")
    new_model.save_weights(weights_path)
    print("Model weights saved!")

    # Method 4: Save as TF checkpoint
    ckpt_path = os.path.join(model_dir, "tf_ckpt")
    ckpt = tf.train.Checkpoint(model=new_model)
    ckpt.save(ckpt_path)
    print(f"Model saved as TF checkpoint to: {ckpt_path}")

except Exception as e:
    print(f"Error during model saving: {e}")
    # Try alternative saving method if the first fails
    try:
        print("Trying alternative saving method...")
        # Create a simple model architecture description
        model_json = new_model.to_json()
        with open(os.path.join(model_dir, "model_architecture.json"), "w") as json_file:
            json_file.write(model_json)
        print("Model architecture saved separately")

        # Save weights only
        new_model.save_weights(os.path.join(model_dir, "model_weights_only.h5"))
        print("Model weights saved separately")
    except Exception as e2:
        print(f"Alternative saving also failed: {e2}")

# Confusion matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot and save confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.tight_layout()
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.savefig(os.path.join(model_dir, "confusion_matrix.png"))
plt.close()

# Save training history
np.save(os.path.join(model_dir, "training_history.npy"), history.history)

print(f"All results saved to {model_dir}")
