const predAPI = require("./src/predictAPI.js");
const express = require("express");
const app = express();

let relativeModelPath = "src/model/tfjs_model/model.json";

app.use(express.json());

app.post("/", (req, res) => {
  let api = new predAPI.predictAPI(
    relativeModelPath,
    req.body.start,
    req.body.destination
  );
  const pred_promise = api.predict();
  pred_promise.then((pred_promise) => {
    var pred = pred_promise;
    const mouseJson = api.parseToJson(pred);
    res.send(mouseJson);
  });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`listening on ${PORT}`));
