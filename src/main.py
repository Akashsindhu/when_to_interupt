import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, LSTM, GRU, ConvLSTM2D, GaussianNoise, Flatten, MaxPool2D, TimeDistributed, Dropout
from tensorflow.keras import Sequential, Input, Model




directory = "../images/"
generator = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
                               validation_split=0.25)
train_gen = generator.flow_from_directory(directory, target_size=(256,256), color_mode='rgb',
                              class_mode="categorical", batch_size=128, shuffle=False, seed=40,
                              subset="training")

test_gen = generator.flow_from_directory(directory, target_size=(256,256), color_mode='rgb',
                              class_mode="categorical", batch_size=128, shuffle=False, seed=40,
                              subset="validation")

print(train_gen.batch_size)
print(test_gen)


inputs = Input(shape=(10,256,256,3))

# convnet = Sequential()
conv2d_layer = Conv2D(filters=64, kernel_size=(3,3))
output = TimeDistributed(conv2d_layer)(inputs)
print(output.shape)
model = Model(inputs=inputs, outputs=output)
# MaxPooling2D(pool_size=2))
# convnet.add(TimeDistributed(Flatten()))
# # convnet.add(TimeDistributed())
# convnet.add(LSTM(128))
# convnet.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

model_LSTM_GRU = Sequential()
model_LSTM_GRU.add(TimeDistributed(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu'), input_shape = (5, 256, 256, 3)))
model_LSTM_GRU.add(TimeDistributed(MaxPool2D(pool_size = (2,2))))
model_LSTM_GRU.add(TimeDistributed(Flatten()))
model_LSTM_GRU.add(LSTM(50, return_sequences = True))
model_LSTM_GRU.add(GRU(50))
model_LSTM_GRU.add(Dense(50, activation = 'relu'))
model_LSTM_GRU.add(Dropout(0.5))
model_LSTM_GRU.add(Dense(50, activation = 'relu'))
model_LSTM_GRU.add(Dense(3, activation = 'softmax'))
model_LSTM_GRU.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

print(model_LSTM_GRU.summary())

# lstm = Sequential()
# lstm.add(TimeDistributed(convnet, input_shape=(128,256,256,3)))
# lstm.add(LSTM(128))
# lstm.add(Dense(3, activation='softmax'))
# lstm.compile(optimizer='adam', loss='mse', metrics='accuracy')
#
# print(lstm.summary())

history = model.fit(train_gen, validation_data=test_gen, verbose=1, epochs=2)
# history.history()










