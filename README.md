# API for Generating Mouse Movements with Neural Networks

Send POST requests to automatically move your mouse with a neural network!

## Table of Contents

- [`pymouse`](#pymouse)
  - [Getting Started](#getting-started)
  - [Dependencies](#dependencies)
  - [Actual Training Pipeline](#actual-training-pipeline)
    - [Path Model](#path-model)
- [JS API](#js-api)
  - [Getting Started (Client)](#getting-started-client)
  - [Model Format Conversion](#model-format-conversion)
  - [Dependencies](#dependencies-1)
  - [How does it work?](#how-does-it-work)

---

**Credits to:**

- [Natural-Mouse-Movements-Neural-Networks](https://github.com/DaiCapra/Natural-Mouse-Movements-Neural-Networks)

---

## pymouse

This is the Python library containing the code for the training pipeline for the models.

The training is mainly done in the Colaboratory notebook. `pymouse` mainly contains some utility functions for callbacks and visualization. The actual training code isn't imported from the module for flexibility, but the training code exists in the library.

Example notebooks are located at [`python/notebooks`](https://github.com/jchen42703/ai_mouse_movements/python/notebooks)

### Getting Started

```
git clone https://github.com/jchen42703/ai_mouse_movements.git
cd python
pip install .
```

### Dependencies

- `numpy`
- `tensorflow`
- `pandas`
- `matplotlib`

### Actual Training Pipeline

#### Path Model

- No preprocessing besides reshaping.
- Training with the path model in `utils.model.py` with `mse`.
  - Start off with `LRFinder` to get the learning rate range and use that range when training again from scratch with the `SGDRScheduler`.
- Predict and you're done!

---

## [JS API]

### Getting Started [Client]

1. Install dependencies with `npm install`
2. `nodemon index.js` or `node index.js` to run the server on `PORT=3000`.
3. Send a `POST` request (`json`) to `"/"`, such as:

```
{
    "start": [1, 1],
    "destination": [82 ,55],
}
```

### Model Format Conversion

#### From `tf.keras` to `.json`

```
pip install tensorflowjs
tensorflowjs_converter --input_format=keras model/weights.h5 model/tfjs_model
```

### Dependencies

- `@tensorflow/tfjs`
- `@tensorflow/tfjs-node`
- `express`
- `body-parser`
- `robotjs` for mouse movements
- `nodemon` for convenience

### How does it work?

1. `POST` request to `https://localhost:3000/`
2. `express` handles the `POST` request and calls the prediction function `loadAndPredict`.
3. The function returns a promise, and the mouse movement (`robotjs`) resolved using this promise.
