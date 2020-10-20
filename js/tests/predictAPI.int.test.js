const predAPI = require("../src/predictAPI.js");
const tf = require("@tensorflow/tfjs");
require("@tensorflow/tfjs-node");

// import predictAPI from "../src/predictAPI.js";
const getMethods = (obj) =>
  Object.getOwnPropertyNames(obj).filter(
    (item) => typeof obj[item] === "function"
  );

test("getRandDest test", () => {
  let relativeModelPath = "../src/model/tfjs_model/model.json";
  let api = new predAPI.predictAPI(relativeModelPath, [5, 300], [600, 500]);
  console.log(`Pred API Instance: ${api}`);
  const dest = api.getRandDest();
  console.log(`Rand Destination: ${dest}`);
  expect(dest.length).toBe(2);
});

test("translate2Origin test", async () => {
  let relativeModelPath = "../src/model/tfjs_model/model.json";
  let api = new predAPI.predictAPI(relativeModelPath, [5, 300], [600, 500]);
  let testTensor = tf.range(0, 3 * 100).reshape([100, 3]);
  let outTensor = api.translate2Origin(testTensor);

  // tests
  expect(outTensor.tensor.shape).toEqual([100, 3]);
  // testing that the first coord is [0, 0, 0]
  let equalsObj = outTensor.tensor
    .slice([0, 0], [1, 3])
    .equal(tf.tensor2d([[0, 0, 0]], [1, 3]));
  // dataSync convert boolean tensor to either 0/1 for false/true
  expect(equalsObj.all().dataSync()[0]).toBe(1);
});

test("scaleCoords test", () => {
  let relativeModelPath = "../src/model/tfjs_model/model.json";
  let api = new predAPI.predictAPI(relativeModelPath, [5, 300], [600, 500]);
  let testTensor = tf.range(0, 3 * 100).reshape([100, 3]);
  let translatedTensor = api.translate2Origin(testTensor);
  // translatedTensor.tensor shape -> [100 ,3]
  const scaled = api.scaleCoords(translatedTensor.tensor, [600, 500]);
  const scaledDest = scaled.slice([-1, 0], [1, 3]);
  console.log(`Scaled Dest Shape: ${scaledDest.shape}`);
  tf.print(scaledDest);
  expect(scaledDest.dataSync()[0]).toEquals(600);
  expect(scaledDest.dataSync()[1]).toEquals(500);
});
