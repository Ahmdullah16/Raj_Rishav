"""TimeDistributed CNN feature extraction followed by sequence LSTM."""
from __future__ import annotations


def build_model(num_classes: int, variant: str = "cnn_lstm_3", seq_length: int = 20,
                image_size: int = 64, learning_rate: float = 1e-3):
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    if num_classes < 2:
        raise ValueError("At least two activity classes are required")
    inputs = layers.Input(shape=(seq_length, image_size, image_size, 3), name="video_rgb")
    x = inputs
    if variant == "mobilenetv2_lstm":
        backbone = tf.keras.applications.MobileNetV2(input_shape=(image_size, image_size, 3),
                                                      include_top=False, weights="imagenet", pooling="avg")
        backbone.trainable = False
        # Pretrained MobileNetV2 expects [-1,1], unlike scratch models.
        x = layers.Rescaling(2.0, offset=-1.0, name="mobilenet_input_range")(x)
        x = layers.TimeDistributed(backbone, name="pretrained_frame_cnn")(x)
        lstm_units = 128
    else:
        if variant == "cnn_lstm_1":
            filters = [32]
            flatten = True
            lstm_units = 64
        elif variant == "cnn_lstm_2":
            filters = [32]
            flatten = False
            lstm_units = 64
        elif variant == "cnn_lstm_3":
            filters = [32, 64, 128]
            flatten = False
            lstm_units = 128
        else:
            raise ValueError(f"Unknown model variant {variant!r}")
        for i, filt in enumerate(filters):
            x = layers.TimeDistributed(layers.Conv2D(filt, 3, padding="same", activation="relu"),
                                       name=f"frame_conv_{i+1}")(x)
            if i < 2:
                x = layers.TimeDistributed(layers.MaxPooling2D(2), name=f"frame_pool_{i+1}")(x)
        x = layers.TimeDistributed(layers.Flatten() if flatten else layers.GlobalAveragePooling2D(),
                                   name="frame_embedding")(x)
    x = layers.LSTM(lstm_units, name="temporal_lstm")(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation="softmax", name="activity_probs")(x)
    model = Model(inputs, output, name=variant)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
