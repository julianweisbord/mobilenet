import tensorflow as tf

IMAGE_SHAPE = (224, 224, 3)
MODEL_SAVE_PATH = "./model/mobilenet.h5"

def save_mobilenet():

    mobilenet_v2 = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SHAPE, weights="imagenet", include_top=False)

    # Add regularization to each layer
    regularizer = tf.keras.regularizers.l2()

    for layer in mobilenet_v2.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # Reload Model with regularization
    model_json = mobilenet_v2.to_json()
    mobilenet_v2 = tf.keras.models.model_from_json(model_json)
    
    # save h5 file
    mobilenet_v2.save(filepath=MODEL_SAVE_PATH, save_format="h5")

if __name__ == '__main__':
    save_mobilenet()
