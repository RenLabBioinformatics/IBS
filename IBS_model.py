import tensorflow as tf
from  tensorflow.keras.metrics import  *
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.layers import Flatten, Dense, Input,Dropout,GlobalAveragePooling2D
from keras.applications.densenet import DenseNet121,DenseNet169
from keras.applications.efficientnet import EfficientNetB0
from keras.applications.mobilenet_v2 import *
from keras.applications.inception_v3 import *
from keras.applications.resnet import *
def Model_VGG16():
    METRICS = [
        tf.keras.metrics.CategoricalAccuracy(name='acc'),
        # tf.keras.metrics.Accuracy(name='acc'),
        tf.keras.metrics.AUC(name='auc', curve='ROC',),
        # tf.keras.metrics.AUC(name='aupr', curve='pr'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=3,name='top3')

    ]
    VGG16_MODEL = tf.keras.applications.VGG16(input_shape=(128,400,3),
                                              include_top=False,
                                              weights='imagenet')
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    dense_layer1 = Dense(256, activation='relu')
    dense_layer2 = Dense(128, activation='relu')
    dropout_Layer = Dropout(0.5)
    prediction_layer = Dense(2, activation='softmax')

    model = tf.keras.Sequential([
        VGG16_MODEL,
        global_average_layer,
        dense_layer1,
        dropout_Layer,
        dense_layer2,
        prediction_layer
    ])
    opt = optimizers.Adam(lr=0.0001)
    model.compile(optimizer=opt,
                  loss=losses.categorical_crossentropy,
                  metrics=METRICS)
    return model

def Model_ResNet():
    METRICS = [
        tf.keras.metrics.Accuracy(name='acc'),
        # tf.keras.metrics.AUC(name='auc', curve='ROC',),
        # tf.keras.metrics.AUC(name='aupr', curve='pr'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
    image_input = Input(shape=(128, 400, 3))
    ResNet50 = tf.keras.applications.ResNet50(input_shape=(128,400,3),
                                              include_top=False,
                                              weights='imagenet')
    global_average_layer = GlobalAveragePooling2D()
    dense_layer1 = Dense(256, activation='relu')
    dense_layer2 = Dense(128, activation='relu')
    dropout_Layer = Dropout(0.8)
    prediction_layer = Dense(2, activation='softmax')

    model = tf.keras.Sequential([
        ResNet50,
        global_average_layer,
        dense_layer1,
        dropout_Layer,
        dense_layer2,
        prediction_layer
    ])
    opt = optimizers.Adam(lr=0.0001)
    model.compile(optimizer=opt,
                  loss=losses.categorical_crossentropy,
                  metrics=METRICS)
    return model

def Model_ResNet_alltype():
    METRICS = [
        tf.keras.metrics.CategoricalAccuracy(name='acc'),
        tf.keras.metrics.AUC(name='auc', curve='ROC',),
        tf.keras.metrics.AUC(name='aupr', curve='pr'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='topK')
    ]
    ResNet50 = tf.keras.applications.ResNet50(input_shape=(256,256,3),
                                              include_top=False,
                                              weights='imagenet')
    # ResNet50.trainable = False
    global_average_layer = GlobalAveragePooling2D()
    dense_layer1 = Dense(128, activation='relu')
    dense_layer2 = Dense(64, activation='relu')
    dropout_Layer = Dropout(0.5)
    prediction_layer = Dense(15, activation='softmax')

    model = tf.keras.Sequential([
        ResNet50,
        global_average_layer,
        # dense_layer1,
        dropout_Layer,
        # dense_layer2,
        prediction_layer
    ])
    opt = optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt,
                  loss=losses.categorical_crossentropy,
                  metrics=METRICS)
    return model
def Model_ResNet_alltype2():
    METRICS = [
        tf.keras.metrics.CategoricalAccuracy(name='acc'),
        tf.keras.metrics.AUC(name='auc', curve='ROC',),
        tf.keras.metrics.AUC(name='aupr', curve='pr'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='topK')
    ]
    ResNet50 = tf.keras.applications.ResNet50(input_shape=(256,256,3),
                                              include_top=False,
                                              weights='imagenet')
    ResNet50.trainable = False
    global_average_layer = GlobalAveragePooling2D()
    dense_layer1 = Dense(256, activation='relu')
    dense_layer2 = Dense(128, activation='relu')
    dropout_Layer = Dropout(0.5)
    prediction_layer = Dense(15, activation='softmax')

    model = tf.keras.Sequential([
        ResNet50,
        global_average_layer,
        dense_layer1,
        dropout_Layer,
        dense_layer2,
        prediction_layer
    ])

    return model
def Model_ResNet_alltype_freeze():
    METRICS = [
        tf.keras.metrics.CategoricalAccuracy(name='acc'),
        tf.keras.metrics.AUC(name='auc', curve='ROC',),
        tf.keras.metrics.AUC(name='aupr', curve='pr'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='topK')
    ]
    ResNet152 = tf.keras.applications.ResNet152(input_shape=(256,256,3),
                                              include_top=False,
                                              weights='imagenet')
    ResNet152.trainable = False
    Efficient = EfficientNetB0(input_shape=(256, 256, 3),
                                                include_top=False,
                                                weights='imagenet')
    Efficient.trainable = False
    mobile = MobileNetV2(input_shape=(256, 256, 3),
                            include_top=False,
                            weights='imagenet')
    mobile.trainable = False
    Efficient.trainable = False
    dense121 = DenseNet121(input_shape=(256, 256, 3),
                         include_top=False,
                         weights='imagenet')
    dense121.trainable = False
    global_average_layer = GlobalAveragePooling2D()
    dense_layer1 = Dense(256, activation='relu')
    dense_layer2 = Dense(128, activation='relu')
    dropout_Layer = Dropout(0.5)
    prediction_layer = Dense(15, activation='softmax')

    model = tf.keras.Sequential([
        ResNet152,
        global_average_layer,
        dense_layer1,
        dropout_Layer,
        dense_layer2,
        prediction_layer
    ])

    return model
def Model_VGG_alltype():
    METRICS = [
        tf.keras.metrics.CategoricalAccuracy(name='acc'),
        tf.keras.metrics.AUC(name='auc', curve='ROC',),
        tf.keras.metrics.AUC(name='aupr', curve='pr'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='topK')
    ]

    VGG19 = tf.keras.applications.VGG19(input_shape=(256,256,3),
                                              include_top=False,
                                              weights='imagenet')
    global_average_layer = GlobalAveragePooling2D()
    dense_layer1 = Dense(256, activation='relu')
    dense_layer2 = Dense(128, activation='relu')
    dropout_Layer = Dropout(0.5)
    prediction_layer = Dense(15, activation='softmax')

    model = tf.keras.Sequential([
        VGG19,
        global_average_layer,
        # dense_layer1,
        dropout_Layer,
        # dense_layer2,
        prediction_layer
    ])
    opt = optimizers.Adam(lr=0.0001)
    model.compile(optimizer=opt,
                  loss=losses.categorical_crossentropy,
                  metrics=METRICS)
    return model
def Model_Dense_alltype():
    METRICS = [
        tf.keras.metrics.CategoricalAccuracy(name='acc'),
        tf.keras.metrics.AUC(name='auc', curve='ROC',),
        tf.keras.metrics.AUC(name='aupr', curve='pr'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='topK')
    ]

    Dense_layer = InceptionV3(input_shape=(256,256,3),
                                              include_top=False,
                                              weights='imagenet')
    global_average_layer = GlobalAveragePooling2D()
    # dense_layer1 = Dense(256, activation='relu')
    # dense_layer2 = Dense(128, activation='relu')
    dropout_Layer = Dropout(0.5)
    prediction_layer = Dense(15, activation='softmax')

    model = tf.keras.Sequential([
        Dense_layer,
        global_average_layer,
        # dense_layer1,
        dropout_Layer,
        # dense_layer2,
        prediction_layer
    ])
    opt = optimizers.Adam(lr=0.0001)
    model.compile(optimizer=opt,
                  loss=losses.categorical_crossentropy,
                  metrics=METRICS)
    return model
def Model_View():
    METRICS = [
        tf.keras.metrics.CategoricalAccuracy(name='acc'),
        tf.keras.metrics.AUC(name='auc', curve='ROC',),
        tf.keras.metrics.AUC(name='aupr', curve='pr'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='topK')
    ]

    Dense_layer = InceptionV3(input_shape=(256,256,3),
                                              include_top=False,
                                              weights='imagenet')
    global_average_layer = GlobalAveragePooling2D()
    dense_layer1 = Dense(256, activation='relu')
    dense_layer2 = Dense(128, activation='relu')
    dropout_Layer = Dropout(0.5)
    prediction_layer = Dense(15, activation='softmax')

    model = tf.keras.Sequential([
        dense_layer1,
        dense_layer2,
        global_average_layer,
        # dense_layer1,
        dropout_Layer,
        # dense_layer2,
        prediction_layer
    ])
    opt = optimizers.Adam(lr=0.0001)
    model.compile(optimizer=opt,
                  loss=losses.categorical_crossentropy,
                  metrics=METRICS)
    return model
if __name__ == '__main__':
    print(tf.test.is_built_with_gpu_support())
    print(tf.test.is_gpu_available())

    # model = Model_ResNet_ZYR()
    METRICS = [
        tf.keras.metrics.CategoricalAccuracy(name='acc'),
        tf.keras.metrics.AUC(name='auc', curve='ROC', ),
        tf.keras.metrics.AUC(name='aupr', curve='pr'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='topK')
    ]
    model = Model_ResNet_alltype2()
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                  loss=losses.categorical_crossentropy,
                  metrics=METRICS)
    model.load_weights("F:\\ImageMatchModel\\all15class\\old\\resnet50\\4fold\\0\\")
    model.summary()