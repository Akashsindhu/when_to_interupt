from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, TimeDistributed, LSTM, Dropout, GRU, Flatten
from keras_video import VideoFrameGenerator

pattern = 'data_path/{classname}/*.avi'

tr_data = VideoFrameGenerator(
          classes = ['confused', 'not_confused', 'uncertain'],
          glob_pattern = pattern,
          nb_frames = 5,
          split_val = 0.5,
          shuffle = False,
          batch_size = 3,
          target_shape = (128, 128),
          nb_channel = 3,
          use_frame_cache = False
)

va_data = tr_data.get_validation_generator()

model_LSTM_GRU = Sequential()
model_LSTM_GRU.add(TimeDistributed(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu'), input_shape = (5, 128, 128, 3)))
model_LSTM_GRU.add(TimeDistributed(MaxPool2D(pool_size = (2,2))))
model_LSTM_GRU.add(TimeDistributed(Flatten()))
model_LSTM_GRU.add(LSTM(50, return_sequences = True))
model_LSTM_GRU.add(GRU(50))
model_LSTM_GRU.add(Dense(50, activation = 'relu'))
model_LSTM_GRU.add(Dropout(0.5))
model_LSTM_GRU.add(Dense(50, activation = 'relu'))
model_LSTM_GRU.add(Dense(3, activation = 'softmax'))
model_LSTM_GRU.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#model_LSTM_GRU.summary()
model_LSTM_GRU.fit_generator(generator = tr_data, validation_data = va_data, verbose = 1, epochs = 50)

