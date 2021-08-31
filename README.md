# API for Generating Mouse Movements with Neural Networks

![](https://github.com/jchen42703/ai_mouse_movements/blob/master/images/mouse_movement_mvp_mouseMove.gif)
Send POST requests to automatically move your mouse with a neural network!

(Note: The api now only generates coordinates and no longer moves the mouse for you.)

## Table of Contents

- [API for Generating Mouse Movements with Neural Networks](#api-for-generating-mouse-movements-with-neural-networks)
  - [Table of Contents](#table-of-contents)
  - [pymousegan](#pymousegan)
    - [Getting Started](#getting-started)
    - [Dependencies](#dependencies)
    - [Training Pipeline](#training-pipeline)
      - [Preprocessing](#preprocessing)
      - [GAN](#gan)
  - [[JS API]](#js-api)
    - [Getting Started [Client]](#getting-started-client)
    - [Model Format Conversion](#model-format-conversion)
      - [From `tf.keras` to `.json`](#from-tfkeras-to-json)
    - [Dependencies](#dependencies-1)
    - [How does it work?](#how-does-it-work)
    - [How fast is the api?](#how-fast-is-the-api)

---

## pymousegan

This is the Python library containing the code for creating neural networks.

The training is done in the Colaboratory notebook. `pymousegan` contains the models and training pipeline for the GAN.

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

### Training Pipeline

#### Preprocessing

1. Translated so that the starting coordinate is `(0, 0)`.
2. Scaled so that the destination coordinates is `(1, 1)`.
3. Reflection across all axes done during training.

#### GAN

The model used in the current version is a `BidirectionalLSTMDecoderGenerator` from an `AdditiveBasicGAN` with a `BidirectionalLSTMDiscriminator` (with minibatch discrimination) and `BidirectionalLSTMDecoderGenerator`. The full example is located at https://github.com/jchen42703/ai_mouse_movements/python/README.md.

Here are the model summaries:

![](images\model_summaries.png)

---

## [JS API]

### Getting Started [Client]

```
cd js
npm install .
nodemon index.js
```

1. Install dependencies with `npm install`
2. `nodemon index.js` or `node index.js` to run the server on `PORT=3000`.
3. Send a `POST` request (`json`) to `http://localhost:3000/`, such as:

```
{
    "start": [1, 1],
    "destination": [82 ,55]
}
```

![](images\mouseMove0.png)

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
- `nodemon` for convenience

### How does it work?

1. `POST` request to `https://localhost:3000/`
2. `express` handles the `POST` request and calls the prediction function `loadAndPredict`.
3. The function returns a promise and when it resolves, the output is a list of coords and lags:

- `[x, y, lag]`
  - The `lag` is the time in `ms` that the mouse stays at that coordinate

```
{
  "coords": [
    [
      1,
      1,
      24.451885223388672
    ],
    [
      1.789207100868225,
      1.6034066677093506,
      23.39274024963379
    ],
    [
      2.462282180786133,
      2.276571035385132,
      24.84036636352539
    ],
    [
      2.7074904441833496,
      2.716768264770508,
      26.283510208129883
    ],
    [
      2.862687110900879,
      3.18359637260437,
      27.842201232910156
    ],
    ...
  ]
}
```

### How fast is the api?

On average, it runs from 390ms to 430ms

- For cold starts: 500ms - 600ms
