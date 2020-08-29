// import * as tf from "@tensorflow/tfjs";

// const model = await tf.loadLayersModel(
//   "https://foo.bar/tfjs_artifacts/model.json"
// );

// const example = tf.fromPixels(webcamElement); // for example
// const prediction = model.predict(example);
import "bootstrap/dist/css/bootstrap.css";
import * as tf from "@tensorflow/tfjs";
import "regenerator-runtime/runtime";

document.getElementById("output").innerText = "Hello World";

(async () => {
  const model = await tf.loadLayersModel(
    "https://drive.google.com/file/d/1mYfiBTxUaSIziWxikQ14TjwfRNnzEPWB/view?usp=sharing"
  );
  model.summary();
})();
// const shape = [1, 2];
// const example = tf.tensor([1, 55], shape);
// console.log(example);
// example.print();
// const prediction = model.predict(example);
