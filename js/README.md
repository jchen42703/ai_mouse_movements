# JS API

## Model Downloads

**You will need to do this section if you plan on reconverting the models to the `tfjs` format.**

- Due to the large size of the `h5` files, I decided to host them elsewhere and to download them.

All provided commands will be with `gdown`:

```
pip install gdown
```

To install the discriminator (`discrim_15001_weights.h5`) into `/src/discrim_model` from `/js`:

```
cd src/discrim_model
gdown https://drive.google.com/uc?id=1c0abQkEHZH4cpGpZxM9AekTLxYPKMWHA
```

To install the generator (`gen_15001_weights.h5`) from `/js`:

```
cd src/model
gdown https://drive.google.com/uc?id=1Jf0GF1picX9lM0ZuDKusfLksIe_RAil1
```

## Model Conversion (`tf.keras` to `.json`)

This assumes that you have the `.h5` models in `discrim_model` for the mouse path testing API or `model` for the mouse path coords generating api.

```
pip install tensorflowjs
cd src
tensorflowjs_converter --input_format=keras model/weights.h5 model/tfjs_model
```

Examples with the current set of weights:

For the generator API:

```
tensorflowjs_converter --input_format=keras model/gen_15001_weights.h5 model/tfjs_model
```

For the mouse path verification API:

```
tensorflowjs_converter --input_format=keras discrim_model/discrim_15001_weights.h5 discrim_model/tfjs_model
```

## How to Use

1. Install dependencies with `npm install`
2. `nodemon index.js` or `node index.js` to run the server on `PORT=3000`.
3. Send a `POST` request (`json`) to `http://localhost:3000/`, such as:

```
{
    "start": [1, 1],
    "destination": [82 ,55]
}
```

## Dependencies

- `@tensorflow/tfjs`
- `@tensorflow/tfjs-node`
- `express`
- `body-parser`
- `robotjs` for mouse movements
- `nodemon` for convenience
