const tf = require("@tensorflow/tfjs");
require("@tensorflow/tfjs-node");

async function loadModel(modelPath) {
  /* Loads the model locally from the relative model path, `modelPath`*/
  const model = await tf.loadLayersModel(`file://${modelPath}`);
  model.summary();
  return model;
}

module.exports = { loadModel };
