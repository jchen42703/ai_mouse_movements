from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Reshape, LSTM
from tensorflow.keras.optimizers import Adam


target_path_count = 100

def lstm_ann(clipvalue=1):
    model = Sequential()
    model.add(Dense(500, activation='relu', input_dim=2))
    model.add(Dense(400, activation='relu'))
    model.add(Dense(target_path_count * 3, activation='linear'))
    model.add(Reshape((target_path_count, 3)))
    model.add(LSTM(300, return_sequences=False, input_shape=(100, 3)))
    model.add(Reshape((target_path_count, 3)))
    opt = Adam(lr=1e-3, clipvalue=clipvalue)
    model.compile(loss='mean_squared_error', optimizer=opt,
                  metrics=['mean_squared_error'])
    return model


def lstm_ann2(clipvalue=1):
    """1.2 million params (training epoch ETA ~17s)
    """
    model = Sequential()
    model.add(Dense(1000, activation='relu', input_dim=2))
    model.add(Dense(400, activation='relu'))
    model.add(Dense(target_path_count * 3, activation='linear'))
    model.add(Reshape((target_path_count, 3)))
    model.add(LSTM(300, return_sequences=False, input_shape=(100, 3)))
    model.add(Reshape((target_path_count, 3)))
    model.add(LSTM(300, return_sequences=False, input_shape=(100, 3)))
    model.add(Reshape((target_path_count, 3)))
    opt = Adam(lr=1e-3, clipvalue=clipvalue)
    model.compile(loss='mean_squared_error', optimizer=opt,
                  metrics=['mean_squared_error'])
    return model


if __name__ == '__main__':
    model = lstm_ann()
    model.summary()
