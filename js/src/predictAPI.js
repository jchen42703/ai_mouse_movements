const tf = require("@tensorflow/tfjs");
const destJson = require("./model/dest.json");

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

predictAPI.prototype.getRandDest = function () {
  // Only gets positive coords
  return destJson[Math.floor(Math.random() * destJson.length)];
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

  console.log(`X scale factor: ${xScale}`);
  console.log(`Y scale factor: ${yScale}`);

  const xyScale = tf.tensor1d([xScale, yScale, 1]);

  let scaled = tf.mul(coords, xyScale);

  return scaled;
};

/**
 * Postprocess (scale API method).
 * @param {tf.Tensor} pred
 * @return {tf.Tensor} path that starts at this.startTensor and ends at
 * this.destTensor
 */
predictAPI.prototype.postprocess = function (pred) {
  pred = pred.squeeze();
  // 1. translate to origin
  // we don't use the predicted offset because we want to guarantee the starting
  // point to `startTensor` rather than to the predicted offset.
  let translated = this.translate2Origin(pred);
  // 2. Scale to (destination + offset)
  let offset = tf.mul(tf.scalar(-1), this.startTensor);
  const newDest = tf.add(this.destTensor, offset);
  let scaled = this.scaleCoords(translated.tensor, newDest);
  // 3. Remove offset from prediction (pred - offset).
  // accomodate for shape (100, 3) w/ (100, 2)->(100, 3)
  offset = tf.tensor2d(
    [offset.slice([0], 1).dataSync()[0], offset.slice([1], 1).dataSync()[0], 0],
    [1, 3]
  );
  console.log(`offset: ${offset}`);
  const out = tf.subtract(scaled, offset);
  return out;
};

predictAPI.prototype.predict = async function () {
  /* Loads the model locally from the relative model path, `modelPath`*/
  const model = await tf.loadLayersModel(`file://${this.modelPath}`);
  let posDest = this.getRandDest();
  let pred = model.predict(posDest);
  pred.print(); // testing
  pred = this.postprocess(pred);
  return pred;
};

module.exports = { predictAPI };
