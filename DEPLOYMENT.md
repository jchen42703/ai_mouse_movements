# Deployment

## From Scratch (DigitalOcean)

````
ssh root@147.182.231.140
sudo ufw enable
ufw allow 80,443,3000,996,7946,4789,2377/tcp; ufw allow 7946,4789,2377/udp
git clone https://github.com/jchen42703/ai_mouse_movements.git

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash

// exit and ssh into root again for nvm to work
ssh root@147.182.231.140

nvm install --lts
cd ai_mouse_movements/js
npm install .

// run as background process
// from 2nd answer of: https://stackoverflow.com/questions/4797050/how-to-run-node-js-as-a-background-process-and-never-die

nohup node index.js > /dev/null 2>&1 &
```

If you encounter a tf error, it will tell you the command to run:

````

sudo apt install build-essential
npm i node-pre-gyp -g
npm rebuild @tensorflow/tfjs-node build-addon-from-source

```

## With Docker

```

DOCKER_BUILDKIT=1 docker build -t jchen42703/ai_mouse_movements .

docker run -dp 3000:80 jchen42703/ai_mouse_movements

// Push to public repo for easy deployment
docker push jchen42703/ai_mouse_movements:latest

```

```
