const express = require("express");
const testPuppeteer = require("./testPuppeteer.js");
const testRobotjs = require("./testRobotjs.js");
const app = express();

// app.use(bodyParser.json());

app.post("/", (req, res) => {
  testPuppeteer.testPuppeteerAssump();
  res.end("Test done!");
});

app.get("/", (req, res) => {
  testRobotjs.testRobotjsAssump();
  res.end("Test done!");
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`listening on ${PORT}`));
