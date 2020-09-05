const predict = require("./src/predict.js");
const path = require("path");
const express = require("express");
const app = express();

let relativeModelPath = "src/model/tfjs_model/model.json";

app.get("/", (req, res) => {
  // res.send(model);
  res.sendFile(path.join(__dirname, relativeModelPath));
  console.log(`dir: ${path.join(__dirname, relativeModelPath)}`);
  console.log(predict.loadModel(relativeModelPath));
  let model = predict.loadModel(relativeModelPath);
  console.log(model);
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`listening on ${PORT}`));
// app.listen(PORT, () => getModel());
