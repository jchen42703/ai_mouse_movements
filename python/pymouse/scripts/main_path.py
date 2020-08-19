from tensorflow.keras.callbacks import ModelCheckpoint
from pymouse.utils.data_loader import load_data
from pymouse.utils.model import init_model_paths
from pymouse.utils.plot import plot

data_path = '/content/Natural-Mouse-Movements-Neural-Networks/train/data.json'
(train_inputs, train_paths, train_time) = load_data(data_path)

# Model
model = init_model_paths()

file_path_best = 'models/path/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(file_path_best, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Fit
history = model.fit(train_inputs, train_paths, epochs=500, verbose=1, validation_split=0.2, callbacks=callbacks_list)

# Plot
plot(model, history, train_inputs, train_paths)
