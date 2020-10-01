robotjs = require("robotjs");

function testRobotjsAssump() {
  //origin
  console.log("Moving to (0, 0)");
  robotjs.moveMouse(0, 0); // starts at top left
  robotjs.setMouseDelay(2000);

  // regular movement
  console.log("Moving to (300, 300)");
  robotjs.moveMouse(500, 300); // 500 to the right and 300 down
  robotjs.setMouseDelay(2000);

  // with negative coords
  console.log("Moving to (-300, -300)");
  robotjs.moveMouse(-300, -300); // to far left corner (neg numbers are a no-no)
  robotjs.setMouseDelay(2000);

  // move with large numbers
  console.log("Moving to (1500, 800)");
  robotjs.moveMouse(1500, 800); // 1500 to the right and 800 down
  obj = robotjs.getScreenSize(); // what is the screen size that robotjs interprets?
  console.log(`Screen size (w x h): ${obj.width} x  ${obj.height}`);
}

module.exports = { testRobotjsAssump };
