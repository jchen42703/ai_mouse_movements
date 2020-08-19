from pymouse.utils.data_loader import load_data
from pymouse.utils.model import init_model_time
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import bar

data_path = '/content/Natural-Mouse-Movements-Neural-Networks/train/data.json'
(train_inputs, train_paths, train_time) = load_data(data_path)

# Model
model = init_model_time()

file_path_best = 'models/time/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(file_path_best, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Fit
history = model.fit(train_inputs, train_time, epochs=1000, verbose=0, validation_split=0.05,
                    callbacks=callbacks_list)

# train_inputs.__len__()
size = 1
for x in range(size):
    values = np.array(train_inputs[x])
    times = model.predict(values.reshape(1, 2))
    flatted = times.flatten()
    i = 0
    for value in flatted:
        plt.bar(i, value)
        i = i + 1

plt.show()
