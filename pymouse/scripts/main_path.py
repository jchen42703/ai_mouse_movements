from keras.callbacks import ModelCheckpoint
import pymouse.utils.data_loader as dl
from pymouse.utils.model import init_model_paths
from pymouse.utils.plot import plot

(train_inputs, train_paths, train_time) = dl.load_data("train/data.json")
# (test_inputs, test_paths) = dl.load_data("test/inputs.csv", "test/paths.json")

# Model
model = init_model_paths()

file_path_best = "models/path/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(file_path_best, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Fit
history = model.fit(train_inputs, train_paths, epochs=1000, verbose=1, validation_split=0.05, callbacks=callbacks_list)

# Plot
plot(model, history, train_inputs, train_paths)
