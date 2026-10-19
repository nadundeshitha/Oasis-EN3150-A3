import tensorflow as tf
from tensorflow.keras import layers, models


def build_model_a(input_shape=(64, 64, 1), num_classes=4):
    """
    Model A (Standard CNN):
    Interleaves standard 2D convolutional layers (Conv2D) and max-pooling layers
    (MaxPooling2D), followed by Flatten, Dense, and Softmax for 4 classes.
    Total trainable parameters: 617,284
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape, name="input_layer"),
            # Block 1: (64, 64, 1) -> (32, 32, 32)
            layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv2d_1"),
            layers.MaxPooling2D((2, 2), name="maxpool_1"),
            # Block 2: (32, 32, 32) -> (16, 16, 64)
            layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2d_2"),
            layers.MaxPooling2D((2, 2), name="maxpool_2"),
            # Block 3: (16, 16, 64) -> (8, 8, 128)
            layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv2d_3"),
            layers.MaxPooling2D((2, 2), name="maxpool_3"),
            # Classification Head
            layers.Flatten(name="flatten"),
            layers.Dense(64, activation="relu", name="fc_1"),
            layers.Dropout(0.3, name="dropout"),
            layers.Dense(num_classes, activation="softmax", name="output_softmax"),
        ],
        name="Model_A_Standard_CNN",
    )
    return model


def build_model_b(input_shape=(64, 64, 1), num_classes=4):
    """
    Model B (Lightweight CNN):
    Replaces standard convolutions with Depthwise Separable Convolutions
    (SeparableConv2D) and Flatten with GlobalAveragePooling2D to drastically
    reduce multiply-accumulate (MAC) operations and keep total trainable
    parameters strictly < 100,000.
    Total trainable parameters: 19,885
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape, name="input_layer"),
            # Block 1: (64, 64, 1) -> (32, 32, 32)
            layers.SeparableConv2D(32, (3, 3), padding="same", activation="relu", name="sep_conv2d_1"),
            layers.MaxPooling2D((2, 2), name="maxpool_1"),
            # Block 2: (32, 32, 32) -> (16, 16, 64)
            layers.SeparableConv2D(64, (3, 3), padding="same", activation="relu", name="sep_conv2d_2"),
            layers.MaxPooling2D((2, 2), name="maxpool_2"),
            # Block 3: (16, 16, 64) -> (8, 8, 128)
            layers.SeparableConv2D(128, (3, 3), padding="same", activation="relu", name="sep_conv2d_3"),
            layers.MaxPooling2D((2, 2), name="maxpool_3"),
            # Lightweight Classification Head
            layers.GlobalAveragePooling2D(name="global_avg_pool"),
            layers.Dense(64, activation="relu", name="fc_1"),
            layers.Dropout(0.3, name="dropout"),
            layers.Dense(num_classes, activation="softmax", name="output_softmax"),
        ],
        name="Model_B_Lightweight_CNN",
    )
    return model


def build_mobilenetv2_sota(input_shape=(64, 64, 1), num_classes=4, fine_tune_at=100):
    """
    Builds a MobileNetV2 transfer-learning model adapted for (64, 64, 1) grayscale input.
    Repeats the single grayscale channel to 3 channels, scales from [0, 1] to [-1, 1]
    as expected by MobileNetV2 ImageNet weights, and attaches a lightweight classifier head.
    """
    inputs = layers.Input(shape=input_shape, name="input_grayscale")
    # Convert (64, 64, 1) -> (64, 64, 3)
    x = layers.Concatenate(axis=-1, name="gray_to_rgb")([inputs, inputs, inputs])
    # MobileNetV2 expects [-1, 1] inputs; our dataset pipeline provides [0, 1]
    x = layers.Rescaling(scale=2.0, offset=-1.0, name="mobilenetv2_rescale")(x)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(input_shape[0], input_shape[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = True
    if fine_tune_at is not None:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output_softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_FineTuned")
    return model


def build_efficientnetb0_sota(input_shape=(64, 64, 1), num_classes=4, fine_tune_at=150):
    """
    Builds an EfficientNetB0 transfer-learning model adapted for (64, 64, 1) grayscale input.
    Repeats the single grayscale channel to 3 channels, scales from [0, 1] to [0, 255]
    as expected by EfficientNetB0's internal Rescaling layer, and attaches a classifier head.
    """
    inputs = layers.Input(shape=input_shape, name="input_grayscale")
    # Convert (64, 64, 1) -> (64, 64, 3)
    x = layers.Concatenate(axis=-1, name="gray_to_rgb")([inputs, inputs, inputs])
    # EfficientNetB0 expects [0, 255] float inputs; our dataset pipeline provides [0, 1]
    x = layers.Rescaling(scale=255.0, offset=0.0, name="efficientnet_rescale")(x)

    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(input_shape[0], input_shape[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = True
    if fine_tune_at is not None:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output_softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_FineTuned")
    return model
