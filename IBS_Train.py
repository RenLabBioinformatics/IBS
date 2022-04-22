from IBS_dataloader import *
from IBS_model import  *
import tensorflow as tf
import gc
import ssl
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score,roc_curve,auc,precision_recall_curve,average_precision_score,confusion_matrix
from sklearn.preprocessing import  label_binarize

import numpy as np

def train():
    ssl._create_default_https_context = ssl._create_unverified_context
    METRICS = [
        tf.keras.metrics.CategoricalAccuracy(name='acc'),
        tf.keras.metrics.AUC(name='auc', curve='ROC', ),
        tf.keras.metrics.AUC(name='aupr', curve='pr'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='topK')
    ]

    # load Training Data
    img_data,Y = load_data()
    print('load data finish!')
    y = np.array(Y)

    skf = StratifiedKFold(n_splits=4,shuffle=True)
    fold_no = 0

    for train_index, test_index in skf.split(img_data, Y):
        # train data
        x_train = img_data[train_index]
        y_train = to_categorical(y[train_index],15)
        # validation data
        x_val = img_data[test_index]
        y_val = to_categorical(y[test_index], 15)
        root_path = 'F:\\IBS_model\\resnet152\\4fold\\'
        save_path = root_path + str(fold_no) + '\\'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            save_weights_only=True,
            monitor='loss',
            mode='min', save_freq='epoch',
            save_best_only=True)
        earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, mode='min', patience=5,
                                                     restore_best_weights=True)
        model = Model_ResNet_alltype_freeze()
        model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                      loss=losses.categorical_crossentropy,
                      metrics=METRICS)

        history = model.fit(x=x_train, y=y_train,
                            batch_size=1,
                            epochs=50, verbose=1, shuffle=True, callbacks=[model_checkpoint_callback,earlystop])
                            # validation_data=(x_val, y_val))
        del(model)
        gc.collect()
        # tf.keras.backend.clear_session()
        print('------------------------------------------------------------------------')
        print(f'Performance of Validation Data for fold {fold_no}')
        top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="top_k_categorical_accuracy", dtype=None)
        top3 = tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_k_categorical_accuracy", dtype=None)
        top5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_k_categorical_accuracy", dtype=None)
        pred_y_val = model.predict(x_val,batch_size = 1)
        top1.update_state(y_true=y_val,y_pred=pred_y_val)
        print('top1 Acc')
        print(top1.result())
        top3.update_state(y_true=y_val, y_pred=pred_y_val)
        print('top3 Acc')
        print(top3.result())
        top5.update_state(y_true=y_val, y_pred=pred_y_val)
        print('top5 Acc')
        print(top5.result())

        auc = roc_auc_score(y_true=y_val, y_score=pred_y_val)
        aupr = average_precision_score(y_true=y_val, y_score=pred_y_val)

        print("Validation AUC  by sklearn: " + str(auc))
        print("Validation AUPR by sklearn: " + str(aupr))

        # cm
        cm = tf.math.confusion_matrix(y_val.argmax(axis=1), pred_y_val.argmax(axis=1),num_classes=15)

        print(cm)
        del(pred_y_val)
        gc.collect()
        tf.keras.backend.clear_session()
        fold_no+=1



if __name__ == '__main__':
    train()
