const tf = require("@tensorflow/tfjs");
require("@tensorflow/tfjs-node");

dataParams = {
  //   minmaxXTrain: [tf.scalar(-2265.0), tf.scalar(2328.0)],
  minmaxYTrain: [tf.scalar(-2265.0), tf.scalar(2328.0)],
  normRange: [tf.scalar(-1), tf.scalar(1)],
};

module.exports = { dataParams };
