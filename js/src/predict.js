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
