from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Reshape, LSTM
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

def lstm_ann_multiple(clipvalue=1):
    """1.2 million (2 outputs: the path and the destination)
    """
    inputs = Input(shape=(2,))
    dense1 = Dense(1000, activation='relu')(inputs)
    dense2 = Dense(400, activation='relu')(dense1)
    dense3 = Dense(target_path_count * 3, activation='linear')(dense2)
    reshaped_path1 = Reshape((target_path_count, 3))(dense3)
    lstm1 = LSTM(300, return_sequences=False,
                 input_shape=(100, 3))(reshaped_path1)
    reshaped_path2 = Reshape((target_path_count, 3))(lstm1)
    lstm2 = LSTM(300, return_sequences=False,
                 input_shape=(100, 3))(reshaped_path2)
    path_out = Reshape((target_path_count, 3), name='path_out')(lstm2)
    dest_out = path_out[-1]
    model = Model(inputs=inputs, outputs=[path_out, dest_out])
    opt = Adam(lr=1e-3, clipvalue=clipvalue)
    model.compile(loss={'path_out': 'mse', 'dest_out': 'mse'}, optimizer=opt)
    return model


def lstm_ann_2mill(clipvalue=1, lr=1e-3):
    """2.5 million params.

    Assumes that the data is ranged from [0, 1] (normally [-1, 1])
    """
    inputs = Input(shape=(2,))
    dense1 = Dense(800, activation='relu')(inputs)
    dense2 = Dense(400, activation='relu')(dense1)
    dense3 = Dense(target_path_count * 3, activation='linear')(dense2)
    reshaped_path1 = Reshape((target_path_count, 3))(dense3)

    lstm1 = LSTM(500, return_sequences=True,
                 input_shape=(100, 3))(reshaped_path1)
    lstm2 = LSTM(300, return_sequences=False)(lstm1)
    dense4 = Dense(target_path_count * 3, activation='linear')(lstm2)
    path_out = Reshape((target_path_count, 3), name='path_out')(dense4)
    model = Model(inputs=inputs, outputs=path_out)

    opt = Adam(lr=lr, clipvalue=clipvalue)
    model.compile(loss='mse', optimizer=opt)
    return model


def lstm_ann_4mill(clipvalue=1, lr=1e-3):
    """4 million params.

    Assumes that the data is ranged from [0, 1] (normally [-1, 1])
    """
    inputs = Input(shape=(2,))
    dense1 = Dense(800, activation='relu')(inputs)
    dense2 = Dense(400, activation='relu')(dense1)
    dense3 = Dense(target_path_count * 3, activation='linear')(dense2)
    reshaped_path1 = Reshape((target_path_count, 3))(dense3)

    lstm1 = LSTM(500, return_sequences=True,
                 input_shape=(100, 3))(reshaped_path1)
    lstm2 = LSTM(300, return_sequences=True, input_shape=(100, 3))(lstm1)
    lstm3 = LSTM(300, return_sequences=True, input_shape=(100, 3))(lstm2)

    lstm4 = LSTM(300, return_sequences=False, input_shape=(100, 3))(lstm3)
    dense4 = Dense(target_path_count * 3, activation='linear')(lstm4)
    path_out = Reshape((target_path_count, 3), name='path_out')(dense4)
    model = Model(inputs=inputs, outputs=path_out)

    opt = Adam(lr=lr, clipvalue=clipvalue)
    model.compile(loss='mse', optimizer=opt)
    return model


if __name__ == '__main__':
    model = lstm_ann_2mill()
    model.summary()
