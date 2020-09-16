from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Reshape, LSTM
from tensorflow.keras.optimizers import Adam


target_path_count = 100

target_path_count = 100

def generate_model():
    model = Sequential()
    model.add(LSTM(128, return_sequences=False, input_shape=(3, 100)))
    #model.add(LSTM(32, return_sequences=True))
    #model.add(LSTM(32))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(target_path_count * 3, activation='linear'))
    model.add(Reshape((target_path_count, 3)))
    # model.summary()
    opt = Adam(lr=1e-3)
    model.compile(loss='mean_squared_error', optimizer=opt,
                  metrics=['mean_squared_error'])
    return model

if __name__ == '__main__':
    model = generate_model()
    model.summary()
