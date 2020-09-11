# JS API

## How to Use

1. Install dependencies with `npm install`
2. `nodemon index.js` or `node index.js` to run the server on `PORT=3000`.
3. Send a `POST` request (`json`) to `"/"`, such as:

```
{
    "start": [1, 1],
    "destination": [82 ,55],
}
```

## Model Conversion (`tf.keras` to `.json`)

```
pip install tensorflowjs
tensorflowjs_converter --input_format=keras model/weights.h5 model/tfjs_model
```

## Dependencies

- `@tensorflow/tfjs`
- `@tensorflow/tfjs-node`
- `express`
- `body-parser`
- `robotjs` for mouse movements
- `nodemon` for convenience
