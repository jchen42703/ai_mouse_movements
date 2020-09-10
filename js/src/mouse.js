robotjs = require("robotjs");

function moveMousePath(pathsTensor) {
  pathsArray = pathsTensor.squeeze().arraySync();
  console.log(`length: ${pathsArray.length}`);
  console.log(`example coord: ${pathsArray[0]}`);
  console.log(`pathsTensor: ${pathsArray}`);
  pathsArray.forEach((coords) => {
    console.log(`coords: ${coords}`);
    robotjs.moveMouse(coords[0], coords[1]);
  });
  //   for (coords in pathsTensor) {
  //     console.log(`coords: ${coords}`);
  //     robotjs.moveMouse(coords[0], coords[1]);
  //   }
  return pathsTensor;
}

module.exports = { moveMousePath };
