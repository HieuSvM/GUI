import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.optimizers import SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from sklearn.preprocessing import LabelEncoder
from keras.regularizers import l2
import keras
import numpy as np
#########################################chia data
y_train=[]
x_train=[]
x=[]
x1=[]
x_test=[]
y_test=[]
y_trainh=[]
y_testh=[]
dem1=0
for f in os.listdir('./training_dataset'):
    dem=0
    for i in os.listdir('./training_dataset/' + f + '/'):
        name_tuple = os.path.splitext(i)
        if (('.txt' in name_tuple[1]))&('name' not in name_tuple[0]):
            if(dem<=(len(os.listdir('./training_dataset/' + f + '/'))-1)*0.7):

                y_train.append(f)
                y_trainh.append(dem1)
            else:
                y_test.append(f)
                y_testh.append(dem1)
            k= open('./training_dataset/' + f + '/'+i, "r")
            line = k.read()
            line = line[0:-1]
            inner_list = [elt.strip() for elt in line.split(',')]
            k.close()
            if (dem <= (len(os.listdir('./training_dataset/' + f + '/'))-1) * 0.7):
                x.append([float(j) for j in inner_list])
            else:
                x1.append([float(j) for j in inner_list])
        dem = dem + 1
    dem1 = dem1 + 1
x_train=np.array(x)
x_test=np.array(x1)
output_dim=len(os.listdir('./training_dataset'))
##############################################################
names_encode = LabelEncoder().fit(y_trainh)
Y = names_encode.transform(y_trainh).tolist()
names_encode1 = LabelEncoder().fit(y_testh)
Y1 = names_encode.transform(y_testh).tolist()
def genDataTrain(x_trainn, y_trainn, output_dimm, batchsize=32):
    len_train = len(x_trainn)
    idx = np.arange(len_train)
    while True:
        np.random.shuffle(idx)
        train_idx = idx[0:len_train]
        x_batch = []
        y_batch = np.zeros((batchsize, output_dimm), dtype=int)
        for i in range(batchsize):
            x_batch.append(x_trainn[train_idx[i]])
            y_batch[i][y_trainn[train_idx[i]]] = 1
        yield np.array(x_batch), np.array(y_batch)
Data = genDataTrain(x_train, Y, output_dim, batchsize=32)
def createFolder(nameFolder):
    try:
        os.makedirs(nameFolder)
    except FileExistsError:
        pass
################################################################# model
seeds = 2
num1 = 512
num2 = 1024

kernel_regul=l2(1e-3)
model_relu = Sequential()

model_relu.add(Dense(num1, activation='relu', input_shape=(128,),
    kernel_regularizer=kernel_regul,
    kernel_initializer=he_normal(seed=seeds)))
model_relu.add(BatchNormalization())
model_relu.add(Dropout(0.5))

model_relu.add(Dense(num2, activation='relu',
    kernel_regularizer=kernel_regul,
    kernel_initializer=he_normal(seed=seeds)))
model_relu.add(BatchNormalization())
model_relu.add(Dropout(0.5))

model_relu.add(Dense(output_dim,activation='softmax'))

print(model_relu.summary())
log_dir = 'logs/'
createFolder(log_dir)
for f in os.listdir(log_dir):
    os.remove(os.path.join(log_dir, f))
model_relu.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
logging = TensorBoard(log_dir=log_dir)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-5)
checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
nb_epoch=300
batch_size=32
####################################################################################

####################################################################################
# model_relu.load_weights('1.model/20210614_1642.h5')


history = model_relu.fit_generator(Data,
                       steps_per_epoch=len(x_train),
                       epochs=nb_epoch,
                       callbacks=[logging, reduce_lr, checkpoint],
                       validation_data=genDataTrain(x_test, Y1, output_dim, batchsize=32),
                       initial_epoch=0,
                       validation_steps=len(x_test)
                       )
#evaluate_generator(self, generator, val_samples)
#results = model_relu.evaluate(genDataTrain(x_test, Y1, output_dim, batchsize=32))
model_relu.save('model.h5',history)
#print('Test loss: {:4f}'.format(results[0]))
#print('Test accuracy: {:4f}'.format(results[1]))
# plot loss và accuracy
# import matplotlib.pyplot as pyplot
# pyplot.figure(figsize=(20,10))
# pyplot.subplot(211)
# pyplot.title('Loss')
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# # plot accuracy during training
# pyplot.subplot(212)
# pyplot.title('Accuracy')
# pyplot.plot(history.history['accuracy'], label='train')
# pyplot.plot(history.history['val_accuracy'], label='test')
# pyplot.legend()
# pyplot.show()