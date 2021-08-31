FROM node:14-alpine
WORKDIR /server
COPY package.json package-lock.json /server/

ENV NODE_ENV=production
RUN npm install --production

COPY . /server
EXPOSE 8080

CMD ["node", "index.js"]