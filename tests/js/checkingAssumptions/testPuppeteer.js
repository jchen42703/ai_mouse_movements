const puppeteer = require("puppeteer");

async function testPuppeteerAssump() {
  const browser = await puppeteer.launch({ headless: false });
  const page = await browser.newPage();
  await page.goto("https://google.com");

  console.log("Moving to (0, 0)");
  await page.mouse.move(0, 0); // starts at top left
  await new Promise((r) => setTimeout(r, 2000));

  // regular movement
  console.log("Moving to (500, 300)");
  await page.mouse.move(500, 300); // 500 to the right and 300 down
  await new Promise((r) => setTimeout(r, 2000));

  // with negative coords
  console.log("Moving to (-300, -300)");
  await page.mouse.move(-300, -300); // to far left corner (neg numbers are a no-no)
  await new Promise((r) => setTimeout(r, 2000));

  // move with large numbers
  console.log("Moving to (1500, 800)");
  await page.mouse.move(1500, 800); // 1500 to the right and 800 down
  await browser.close();
}

module.exports = { testPuppeteerAssump };
