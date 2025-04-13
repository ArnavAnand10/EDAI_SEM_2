import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Model, Sequential, optimizers
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization,
    Activation,
    Conv2D,
    SeparableConv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    ConvLSTM2D,
    TimeDistributed,
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    ReduceLROnPlateau,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import config

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
print("Loading data...")
XAuthenticate = list(np.load(config.np_casia_two_au_path))
yAuthenticate = list(np.zeros(len(XAuthenticate), dtype=np.uint8))
XForged = list(np.load(config.np_casia_two_forged_path))
yForged = list(np.ones(len(XForged), dtype=np.uint8))
X = np.array(XAuthenticate + XForged)
y = np.array(yAuthenticate + yForged, dtype=np.int8)
print("Total data:", X.shape, "Labels:", y.shape)
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, shuffle=True, random_state=42
)
plt.hist(y, bins=5)
plt.ylabel("Number of images")
plt.title("CASIA II - Authenticate OR Fake")
plt.savefig("class_distribution.png")
plt.close()
img_height = 256
img_width = 384


def xceptionNet():
    inp = Input(shape=(img_height, img_width, 3))
    x = Conv2D(
        32,
        (3, 3),
        strides=(2, 2),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
    )(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(
        64,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    residual = Conv2D(
        128,
        (1, 1),
        strides=(2, 2),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-2),
    )(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.add([x, residual])
    residual = Conv2D(
        256,
        (1, 1),
        strides=(2, 2),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-2),
    )(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(
        256,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(
        256,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.add([x, residual])
    residual = Conv2D(
        728,
        (1, 1),
        strides=(2, 2),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-2),
    )(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(
        728,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(
        728,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.add([x, residual])
    for _ in range(8):
        residual = x
        x = Activation("relu")(x)
        x = SeparableConv2D(
            728,
            (3, 3),
            padding="same",
            depthwise_initializer="he_normal",
            pointwise_initializer="he_normal",
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = SeparableConv2D(
            728,
            (3, 3),
            padding="same",
            depthwise_initializer="he_normal",
            pointwise_initializer="he_normal",
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = SeparableConv2D(
            728,
            (3, 3),
            padding="same",
            depthwise_initializer="he_normal",
            pointwise_initializer="he_normal",
        )(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.add([x, residual])
    residual = Conv2D(
        1024,
        (1, 1),
        strides=(2, 2),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-2),
    )(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(
        728,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(
        1024,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.add([x, residual])
    x = SeparableConv2D(
        1536,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(
        2048,
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    out = Dense(2, activation="softmax", kernel_initializer="he_normal")(x)
    return Model(inp, out)


def adjust_lr(epoch):
    base = 1e-2
    if epoch > 160:
        return base * 5e-4
    elif epoch > 120:
        return base * 1e-3
    elif epoch > 80:
        return base * 5e-3
    elif epoch > 40:
        return base * 5e-2
    else:
        return base * 1e-1


print("Building and training Xception model")
xcept_model = xceptionNet()
xcept_model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.Adam(learning_rate=adjust_lr(0)),
    metrics=["accuracy"],
)
save_dir = os.path.join(os.getcwd(), "saved_models")
os.makedirs(save_dir, exist_ok=True)
xcept_ckpt = os.path.join(save_dir, "xception_best.h5")
xcept_cb = ModelCheckpoint(
    filepath=xcept_ckpt, monitor="val_accuracy", save_best_only=True, verbose=1
)
lr_red = ReduceLROnPlateau(factor=np.sqrt(0.1), patience=5, min_lr=5e-6, verbose=1)
lr_sched = LearningRateScheduler(adjust_lr, verbose=1)
y_cat = to_categorical(y, 2)
history1 = xcept_model.fit(
    x_train,
    y_cat,
    batch_size=32,
    epochs=5,
    validation_data=(x_test, to_categorical(y_test, 2)),
    callbacks=[xcept_cb, lr_red, lr_sched],
)
print("Evaluating Xception model")
best_xcept = load_model(xcept_ckpt)
yp_x = best_xcept.predict(x_test)
print(
    classification_report(
        np.argmax(to_categorical(y_test, 2), axis=1), np.argmax(yp_x, axis=1)
    )
)
print("Extracting features")
feature_model = Model(inputs=best_xcept.inputs, outputs=best_xcept.layers[-4].output)
features_train = feature_model.predict(x_train)
features_test = feature_model.predict(x_test)
timesteps = 5
seq_train = np.expand_dims(features_train, axis=1)
seq_train = np.tile(seq_train, (1, timesteps, 1, 1, 1))
seq_test = np.expand_dims(features_test, axis=1)
seq_test = np.tile(seq_test, (1, timesteps, 1, 1, 1))
print("Building ConvLSTM model")
lstm_model = Sequential()
lstm_model.add(
    TimeDistributed(
        Conv2D(128, (1, 1), activation="relu"),
        input_shape=(
            timesteps,
            features_train.shape[1],
            features_train.shape[2],
            features_train.shape[3],
        ),
    )
)
lstm_model.add(
    ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(2, 2),
        kernel_initializer="he_normal",
        return_sequences=False,
    )
)
lstm_model.add(Flatten())
lstm_model.add(Dense(256, activation="relu"))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(2, activation="softmax"))
lstm_model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
lstm_ckpt = os.path.join(save_dir, "lstm_best.h5")
lstm_cb = ModelCheckpoint(
    filepath=lstm_ckpt, monitor="val_accuracy", save_best_only=True, verbose=1
)
history2 = lstm_model.fit(
    seq_train,
    to_categorical(y_train, 2),
    batch_size=8,
    epochs=5,
    validation_data=(seq_test, to_categorical(y_test, 2)),
    callbacks=[lstm_cb],
)
print("Evaluating ConvLSTM model")
best_lstm = load_model(lstm_ckpt)
yp_lstm = best_lstm.predict(seq_test, batch_size=8)
print(
    classification_report(
        np.argmax(to_categorical(y_test, 2), axis=1), np.argmax(yp_lstm, axis=1)
    )
)
final_model_path = os.path.join(save_dir, "casia2_model.h5")
best_lstm.save(final_model_path)
print("Final model saved to", final_model_path)
plt.figure(figsize=(8, 6))
cm = confusion_matrix(
    np.argmax(to_categorical(y_test, 2), axis=1), np.argmax(yp_lstm, axis=1)
)
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.tight_layout()
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
plt.close()
np.save(
    os.path.join(save_dir, "training_history.npy"),
    {"xception": history1.history, "lstm": history2.history},
)
print("Everything saved in", save_dir)
