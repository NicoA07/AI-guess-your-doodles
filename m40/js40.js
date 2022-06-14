const IMAGE_SIZE = 784;
const CLASSES = [
    'banana', 
    'bear', 
    'bee', 
    'bird',  
    'book', 
    'carrot', 
    'cat', 
    'cell_phone', 
    'circle', 
    'dog', 
    'duck',
    'elephant', 
    'grapes', 
    'house', 
    'key', 
    'ladder', 
    'laptop', 
    'moon',
    'mug',
    'octagon',
    'panda',
    'pear',
    'peas',
    'pineapple',
    'sword',
    'rainbow',
    'shoe',
    'smiley_face',
    'snowman',
    'spider',
    'square',
    'star',
    'strawberry',
    'sun',
    'tiger',
    'tree',
    'triangle',
    'washing machine',
    'watermelon',
    'windmill']

const k = 5;
let model;
let cnv;

async function loadMyModel() {
  model = await tf.loadLayersModel('m40/model40/model.json');
  model.summary();
}

function setup() {
  loadMyModel();

  cnv = createCanvas(280, 280);
  background(255);
  cnv.mouseReleased(guess);

  let clearButton = select('#clear');
  clearButton.mouseReleased(() => {
    background(255);
    select('#res').html('');
  });
}

function guess() {
  const inputs = getInputImage();
  let guess = model.predict(tf.tensor([inputs]));
  const rawProb = Array.from(guess.dataSync());
  const rawProbWIndex = rawProb.map((probability, index) => {
    return {index,probability}
  });

  const sortProb = rawProbWIndex.sort((a, b) => b.probability - a.probability);
  const topKClassWIndex = sortProb.slice(0, k);
  const topKRes = topKClassWIndex.map(i => `<br>${CLASSES[i.index]} (${(i.probability.toFixed(2) * 100)}%)`);
  select('#res').html(`Your doodle is ${topKRes.toString()}`);
}

function getInputImage() {
  let inputs = [];
  let img = get();
  img.resize(28, 28);
  img.loadPixels();

  let oneRow = [];
  for (let i = 0; i < IMAGE_SIZE; i++) {
    let bright = img.pixels[i * 4];
    let onePix = [parseFloat((255 - bright) / 255)];
    oneRow.push(onePix);
    if (oneRow.length === 28) {
      inputs.push(oneRow);
      oneRow = [];
    }
  }
  return inputs;
}

function draw() {
  strokeWeight(10);
  stroke(0);
  if (mouseIsPressed) {
    line(pmouseX, pmouseY, mouseX, mouseY);
  }
}