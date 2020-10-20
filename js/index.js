const predAPI = require("./src/predictAPI.js");
const mouse = require("./src/mouse.js");
const path = require("path");
const bodyParser = require("body-parser");
const express = require("express");
const app = express();

let relativeModelPath = "src/model/tfjs_model/model.json";

app.use(bodyParser.json());

app.post("/", (req, res) => {
  console.log(req.body);
  console.log(`dir: ${path.join(__dirname, relativeModelPath)}`);
  let api = new predAPI.predictAPI(
    relativeModelPath,
    req.body.start,
    req.body.destination
  );
  const pred_promise = api.predict();
  pred_promise.then((pred_promise) => {
    var pred = pred_promise;
    console.log(`prediction: ${pred}, shape: ${pred.shape}`);
    // move mouse....
    mouse.moveMousePath(pred);
  });
  res.end("Movement done!");
});

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, relativeModelPath));
  res.end("File sent!");
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`listening on ${PORT}`));
