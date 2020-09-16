from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.optimizers import Adam


target_path_count = 100


def init_model_paths_large():
    """Model for predicting paths.
    5,409,700 params.
    """
    model = Sequential()
    model.add(Dense(2500, activation='relu', input_dim=2))
    model.add(Dense(2000, activation='relu'))
    model.add(Dense(target_path_count * 2, activation='linear'))
    model.add(Reshape((target_path_count, 2)))

    # model.summary()
    opt = Adam(lr=1e-3)
    model.compile(loss='mean_squared_error', optimizer=opt,
                  metrics=['mean_squared_error'])
    return model


def init_model_paths():
    """Original paths model.
    203,200 params.
    """
    model = Sequential()
    model.add(Dense(1000, activation='relu', input_dim=2))
    model.add(Dense(target_path_count * 2, activation='linear'))
    model.add(Reshape((target_path_count, 2)))

    # model.summary()
    opt = Adam(lr=1e-4)
    model.compile(loss='mean_squared_error', optimizer=opt,
                  metrics=['mean_squared_error', 'accuracy'])

    return model


def init_model_time():
    model = Sequential()
    model.add(Dense(216, activation='relu', input_dim=2))
    model.add(Dense(target_path_count, activation='linear'))
    model.add(Reshape((target_path_count, 1)))

    # model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error', 'accuracy'])

    return model
