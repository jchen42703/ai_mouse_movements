# Deployment

## With Docker

```
DOCKER_BUILDKIT=1 docker build -t jchen42703/ai_mouse_movements .

docker run -dp 3000:80 jchen42703/ai_mouse_movements

// Push to public repo for easy deployment
docker push jchen42703/ai_mouse_movements:latest
```
