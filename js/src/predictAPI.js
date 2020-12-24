const tf = require("@tensorflow/tfjs");

require("@tensorflow/tfjs-node");

/** 
 Clean for single predictions, but don't use for multiple predictions.
* @param {string} modelPath relative model path (relative to `index.js`)
* @param {array} start [X_start, Y_start]
* @param {array} destination [X_dest, Y_dest]
*/
function predictAPI(modelPath, start, destination) {
  this.modelPath = modelPath;
  // convert to tensors
  this.startTensor = tf.tensor2d(start, [1, 2]);
  this.destTensor = tf.tensor2d(destination, [1, 2]);
}

/**
 * Generates the random noise for the generator. (normal distribution)
 * @param {array} noiseSize should be [batch_size, 100, num_latent_dims]
 * @return {tf.Tensor} the random noise with the shape, noiseSize
 */
predictAPI.prototype.getRandNoise = function (noiseSize) {
  return tf.randomNormal(noiseSize, 0, 1);
};

/**
 * Translates coordinates such that the starting coordinate is @ (0, 0, 0).
 * @param {tf.Tensor} coords with shape (path_count, 3)).
 * Assumes that all of the coords >= 0.
 * @return {obj{tf.Tensor, tf.Tensor}} the translated coords (tensor) and the offset
 * offset is defined as the (-1) * start, so start + offset = (0, 0, 0).
 * Therefore, (offsetted predictions) - offset leads to the proper prediction.
 */
predictAPI.prototype.translate2Origin = function (coords) {
  // grabbing 1st coord of coords
  const coords1 = coords.slice([0, 0], [1, 3]);
  const offset = tf.mul(tf.scalar(-1), coords1);
  const tensor = tf.add(coords, offset);
  return { tensor, offset };
};

/**
 * Scales the coordinates so that the destination of `coords` matches
 * `dest`.
 * @param {tf.Tensor} coords with shape (path_count, 3)).
 * Assumes that all of the coords >= 0 and start at the origin.
 * @param {array} dest [X, Y] coordinate for the desired destination.
 * @return {tf.Tensor} the properly scaled coordinates.
 */
predictAPI.prototype.scaleCoords = function (coords, dest) {
  // Ratio desired_destination / coord_destination
  const lastCoord = coords.slice([99, 0], [1, 3]).dataSync();

  const xScale = dest[0] / lastCoord[0];
  const yScale = dest[1] / lastCoord[1];

  // console.log(`X scale factor: ${xScale}`);
  // console.log(`Y scale factor: ${yScale}`);

  const xyScale = tf.tensor1d([xScale, yScale, 1]);

  let scaled = tf.mul(coords, xyScale);

  return scaled;
};

/**
 * Postprocess (scale API method).
 * @param {tf.Tensor} pred
 * @return {tf.Tensor} path that starts at this.startTensor and ends at
 * this.destTensor and a time distribution; shape: (100, 3)
 */
predictAPI.prototype.postprocess = function (pred) {
  pred = pred.squeeze();

  const dt = this.postprocessDT(pred);
  // 1. translate to origin
  // we don't use the predicted offset because we want to guarantee the starting
  // point to `startTensor` rather than to the predicted offset.
  let translated = this.translate2Origin(pred);
  // 2. Scale to (destination + offset)
  let offset = tf.mul(tf.scalar(-1), this.startTensor);
  const newDest = tf.add(this.destTensor, offset).dataSync();
  let scaled = this.scaleCoords(translated.tensor, newDest);
  // 3. Remove offset from prediction (pred - offset).
  // accomodate for shape (100, 3) w/ (100, 2)->(100, 3)
  const offsetArr = offset.dataSync();
  offset = tf.tensor2d([[offsetArr[0], offsetArr[1], 0]], [1, 3]);
  const out = tf.sub(scaled, offset);
  return out.concat(dt, (axis = 1));
};

/**
 *
 * @param {tf.Tensor} pred squeezed prediction tensor with shape [100, 3]
 * @returns {tf.Tensor} 1d tensor of times
 */
predictAPI.prototype.getDT = function (pred) {
  const unstacked = tf.unstack(pred);
  let dt = [];
  unstacked.forEach((tensor) => dt.push(tensor.slice(2, 1).arraySync()[0]));
  return tf.expandDims(tf.tensor1d(dt), (axis = -1));
};

predictAPI.prototype.postprocessDT = function (squeezedPred) {
  let dt = this.getDT(squeezedPred);
  const min = dt.min();
  // console.log(`dt min ${min.dataSync()}`);
  if (min.dataSync() < 0) {
    dt = tf.add(dt, min.mul(-1));
  }
  return tf.mul(dt, 250);
};

predictAPI.prototype.predict = async function () {
  /* Loads the model locally from the relative model path, `modelPath`*/
  const model = await tf.loadLayersModel(`file://${this.modelPath}`);
  let randNoise = this.getRandNoise([1, 100, 100]);
  let pred = model.predict(randNoise);
  pred.print(); // testing
  pred = this.postprocess(pred);
  return pred;
};

module.exports = { predictAPI };
