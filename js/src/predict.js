const tf = require("@tensorflow/tfjs");
require("@tensorflow/tfjs-node");

async function loadModel(modelPath) {
  /* Loads the model locally from the relative model path, `modelPath`*/
  const model = await tf.loadLayersModel(`file://${modelPath}`);
  model.summary();
  return model;
}

function getModelPred(pred_model, start, destination) {
  /** 
  Gets the predicted path based on the origin and destination
  * @param {tf.model} pred_model 
  * @param {array} start [X_start, Y_start]
  * @param {array} destination [X_dest, Y_dest]
  * @return {tf.Tensor} pred with shape (1, 100, 2)
  */
  // convert to tensors
  var start = tf.tensor2d(start, [1, 2]);
  var dest = tf.tensor2d(destination, [1, 2]);
  // normalize the starting point to origin
  // so that start + offset = [0, 0] & pred (assumes [0, 0])
  // so pred - offset = pred assuming start as the starting point
  const offset = tf.mul(tf.scalar(-1), start);
  dest = tf.add(dest, offset);
  var pred = pred_model.predict(dest);
  pred.print(); // testing
  pred = tf.sub(pred, offset);
  return pred;
}

function minmaxNormalize(
  arr,
  normRange = [tf.scalar(-1), tf.scalar(1)],
  minmax = null
) {
  /**
   * Normalizes the input array between a specified range.
   * @param {tf.Tensor} arr input array to normalize
   * @param {array-like} normRange array of 2 integers specifying normalizing range
   * @param {array-like} minmax array specifying the minimum and maximum of `arr`. If left as null,
   the minmax is automatically calculated.
   * @return {tf.Tensor} the normalized tensor
   */
  if (minmax == null) {
    var min = arr.min();
    var max = arr.max();
  } else {
    var min = minmax[0];
    var max = minmax[1];
  }
  const scalingFactor = normRange[1].sub(normRange[0]);
  const regNorm = tf.div(tf.sub(arr, min), tf.sub(max, min));
  var normArr = tf.mul(scalingFactor, regNorm).add(normRange[0]);
  return normArr;
}

function minmaxUnnormalize(
  normArr,
  minmax,
  normRange = [tf.scalar(-1), tf.scalar(1)]
) {
  /**
   * Undos minmaxNormalize.
   * @param {tf.Tensor} normArr input array to normalize
   * @param {array-like of tf.scalar} minmax array specifying the minimum and maximum of `arr`. If left as null,
    the minmax is automatically calculated.
   * @param {array-like of tf.scalar} normRange array of 2 integers specifying normalizing range
   * @return {tf.Tensor} the normalized tensor
   */
  const min = minmax[0];
  const max = minmax[1];
  const scalingFactor = normRange[1].sub(normRange[0]);
  const maxMinDiff = max.sub(min);

  // arr = ((norm_arr - norm_range[0]) / (norm_range[1]-norm_range[0]) * (max - min)) + min
  var arr = tf.add(
    tf.mul(tf.div(tf.sub(normArr, normRange[0]), scalingFactor), maxMinDiff),
    min
  );
  return arr;
}

async function loadAndPredict(modelPath, start, destination) {
  /** 
   Clean for single predictions, but don't use for multiple predictions.
   * @param {string} modelPath relative model path (relative to `index.js`)
   * @param {tf.model} pred_model 
   * @param {array} start [X_start, Y_start]
   * @param {array} destination [X_dest, Y_dest]
   * @return {tf.Tensor} pred with shape (1, 100, 2)
   */
  /* Loads the model locally from the relative model path, `modelPath`*/
  const model = await tf.loadLayersModel(`file://${modelPath}`);
  // convert to tensors
  var start = tf.tensor2d(start, [1, 2]);
  var dest = tf.tensor2d(destination, [1, 2]);
  // normalize the starting point to origin
  // so that start + offset = [0, 0] & pred (assumes [0, 0])
  // so pred - offset = pred assuming start as the starting point
  const offset = tf.mul(tf.scalar(-1), start);
  dest = tf.add(dest, offset);
  var pred = model.predict(dest);
  pred.print(); // testing
  pred = tf.sub(pred, offset);
  return pred;
}

function offsetCoords(start, destination) {
  /* For testing the offsetting + shapes */
  // convert to tensors
  var start = tf.tensor2d(start, [1, 2]);
  var dest = tf.tensor2d(destination, [1, 2]);
  // normalize to origin
  // so that start + offset = [0, 0] & pred (assumes [0, 0])
  // so pred - offset = pred assuming start as the starting point
  const offset = tf.mul(tf.scalar(-1), start);
  dest = tf.add(dest, offset);
  return offset;
}

module.exports = { loadModel, getModelPred, loadAndPredict };
