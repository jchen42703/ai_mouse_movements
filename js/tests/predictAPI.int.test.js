const predAPI = require("../src/predictAPI.js");
const tf = require("@tensorflow/tfjs");
require("@tensorflow/tfjs-node");

// import predictAPI from "../src/predictAPI.js";
const getMethods = (obj) =>
  Object.getOwnPropertyNames(obj).filter(
    (item) => typeof obj[item] === "function"
  );

test("getRandDest test", () => {
  let relativeModelPath = "./src/model/tfjs_model/model.json";
  let api = new predAPI.predictAPI(relativeModelPath, [5, 300], [600, 500]);
  console.log(`Pred API Instance: ${api}`);
  const dest = api.getRandDest();
  console.log(`Rand Destination: ${dest}`);
  expect(dest.length).toBe(2);
});

test("translate2Origin test", async () => {
  let relativeModelPath = "./src/model/tfjs_model/model.json";
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
  let relativeModelPath = "./src/model/tfjs_model/model.json";
  let api = new predAPI.predictAPI(relativeModelPath, [5, 300], [600, 500]);
  let testTensor = tf.range(0, 3 * 100).reshape([100, 3]);
  let translatedTensor = api.translate2Origin(testTensor);
  // translatedTensor.tensor shape -> [100 ,3]
  const scaled = api.scaleCoords(translatedTensor.tensor, [600, 500]);
  const scaledDest = scaled.slice([99, 0], [1, 3]);

  expect(scaledDest.dataSync()[0]).toEqual(600);
  expect(scaledDest.dataSync()[1]).toEqual(500);
});

test("postprocess test (+) (+)", async () => {
  /* Loads the model locally from the relative model path, `modelPath`*/
  // Note: Running test at ai_mouse_movements/js
  let relativeModelPath = "./src/model/tfjs_model/model.json";
  let api = new predAPI.predictAPI(relativeModelPath, [5, 300], [600, 500]);
  const model = await tf.loadLayersModel(`file://${relativeModelPath}`);
  let posDest = api.getRandDest();
  let pred = model.predict(tf.tensor2d(posDest, [1, 2]));

  pred = api.postprocess(pred).squeeze();

  console.log(`pred shape (after squeeze): ${pred.shape}`);

  const firstCoord = pred.slice([0, 0], [1, 3]).dataSync();
  const lastCoord = pred.slice([99, 0], [1, 3]).dataSync();

  console.log(`First coord: ${firstCoord}, last: ${lastCoord}`);
  expect(firstCoord[0]).toBe(5);
  expect(firstCoord[1]).toBe(300);

  expect(lastCoord[0]).toBe(600);
  expect(lastCoord[1]).toBe(500);
});
