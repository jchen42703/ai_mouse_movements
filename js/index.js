const predict = require("./src/predict.js");
const path = require("path");
const express = require("express");
const app = express();

let relativeModelPath = "src/model/tfjs_model/model.json";

app.get("/", (req, res) => {
  // res.send(model);
  res.sendFile(path.join(__dirname, relativeModelPath));
  console.log(`dir: ${path.join(__dirname, relativeModelPath)}`);
  const pred_promise = predict.loadAndPredict(
    relativeModelPath,
    [1, 1],
    [82, 33]
  );
  pred_promise.then((pred_promise) => {
    var pred = pred_promise;
    console.log(`prediction: ${pred}`);
    // move mouse....
  });

  /**
   * predict.loadModel -> predict.getModelPred pipeline
   * awkward, but more readable
   */
  // let model = predict.loadModel(relativeModelPath);
  // const pred = model.then((model) => {
  //   predict.getModelPred(model, [1, 1], [82, 33]);
  // });
  // console.log(`Final prediction: ${pred}`);
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`listening on ${PORT}`));
// app.listen(PORT, () => getModel());
