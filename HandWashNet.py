from os import path
import numpy as np
import talos
from numpy import save, load
from sklearn.model_selection import train_test_split
import os.path
import HandWashNet1.dbGenerate as db

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto(allow_soft_placement=False)
config.gpu_options.per_process_gpu_memory_fraction = 0.9
# config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def modelStandard(x_train, y_train, x_val, y_val, params):
    from HandWashNet1.HandWashModels import modelStandard
    model = modelStandard(input_shape=(None, SIZE, SIZE, 1), parameter=params);

    print(model.summary())
    from tensorflow.keras.utils import plot_model
    plot_model(model, show_shapes=True, to_file=ROOTPATH +CURVERSION+ params['Name'] + '_model.png')

    # if we want to also test for number of layers and shapes, that's possible
    # hidden_layers(model, params, 1)

    history = model.fit(x_train, y_train, batch_size=params['batch_size'],
                        epochs=params['epochs'], validation_split=0.01,
                        shuffle=False,
                        workers=2,
                        use_multiprocessing=True)

    # evaluate on new data
    loss, acc = model.evaluate(x_val, y_val, verbose=0, batch_size=BATCHSIZE)
    print('loss: %f, acc: %f' % (loss, acc * 100))

    # prediction on new data
    yhat = model.predict(x_val, verbose=0, batch_size=BATCHSIZE)
    expected = [np.argmax(y_val, axis=1, out=None) for y in Y]
    predicted = predicted = np.argmax(yhat, axis=1)
    print('Expected: %s, Predicted: %s ' % (expected, predicted))


    # finally we have to make sure that history object and model are returned
    return history, model




if __name__ == "__main__":
    tag = {0: 'NailWashLeft', 1: 'NailWashRight', 2: 'ThumbFingureWash', 3: 'ForeFingureWash'}
    inv_tag = {v: k for k, v in tag.items()}

    ROOTPATH = '/data1/shengjun/HandWash/'
    CURVERSION='v1/'
    DB = 'db/'

    SAMPLESNUM = 3000
    DBNUM=str(SAMPLESNUM)
    ITERATE = 1
    SIZE = 50
    SHUFF = True
    BATCHSIZE = 8

    p = {'modelStandard': {'Name': ['modelStandard'],
                           'shuff': [False],
                           'activation': ['relu', 'elu'],
                           'optimizer': ['AdaDelta','Adam'],
                           'losses': ['logcosh'],
                           'shapes': ['brick'],
                           'first_neuron': [32],
                           'dropout': [.2, .3],
                           'cell1':[40,50,60,70,80],
                           'cell2': [30, 40, 50, 60, 70],
                           'batch_size': [4,8,16,32],
                           'num_layers': [5],
                           'output_activation': ['sigmoid'],
                           'filters': [64],
                           'epochs': [1]},
         }
    fileNameTrain=''
    fileNameLabel=''
    if p['modelStandard']['shuff'] ==True:
        fileNameTrain=ROOTPATH + DB + 'dba_train_shuff' + DBNUM + '.npy'
        fileNameLabel=ROOTPATH + DB + 'dba_label_shuff' + DBNUM + '.npy'
        if path.exists(fileNameTrain) and path.exists(fileNameLabel):
            X = load(fileNameTrain)
            y = load()
        else:
            X, y = db.generate_DB_A(size=SIZE, n_patterns=SAMPLESNUM, parameter=p['modelStandard'])
            save(fileNameTrain, X)
            save(fileNameLabel, y)
    else:
        fileNameTrain = ROOTPATH + DB + 'dba_train' + DBNUM + '.npy'
        fileNameLabel = ROOTPATH + DB + 'dba_label' + DBNUM + '.npy'
        if path.exists(fileNameTrain) and path.exists(fileNameLabel):
            X = load(fileNameTrain)
            y = load(fileNameLabel)
        else:
            X, y = db.generate_DB_A(size=SIZE, n_patterns=SAMPLESNUM, parameter=p['modelStandard'])
            save(fileNameTrain, X)
            save(fileNameLabel, y)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
    print("scan modelStandard")
    t = talos.Scan(x=X_train, y=y_train, x_val=X_test, y_val=y_test, params=p['modelStandard'], model=modelStandard,
                   reduction_metric='val_iou', experiment_name='modelStandard')
