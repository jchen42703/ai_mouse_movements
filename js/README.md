# JS API

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
