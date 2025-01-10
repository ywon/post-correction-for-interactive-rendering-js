#/bin/bash

# Stop previous container and remove it
docker rm $(docker stop $(docker ps -a -q --filter ancestor=post-correction-for-interactive-rendering-js --format="{{.ID}}"))

# Remove previous image
docker rmi post-correction-for-interactive-rendering-js

# Build docker image
# docker build --no-cache --build-arg UID=$(id -u) --build-arg GID=$(id -g) . -t rt-denoiser-simple
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) . -t post-correction-for-interactive-rendering-js

# Launch a container
docker run --privileged --gpus all -dit -v `pwd`:/home/pcir/post-correction-for-interactive-rendering-js post-correction-for-interactive-rendering-js


