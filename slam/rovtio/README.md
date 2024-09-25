# ROVTIO Docker Image
## Installation 
Clone the repo
```
git clone https://github.com/rongirl/thermal-slam-dockers.git 
cd slam/rovtio
git submodule update --init --recursive
```
## Launch roscore
```
roscore
```
## Building Docker Image
Build docker image using `Dockerfile`:
```
docker build -t rovtio_image -f Dockerfile ..
```
## Running Docker Container
To run the container use the following command:
```
docker run --net=host  -it --env DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix rovtio
```
## Launch the rosbag and rviz
```
rviz
```
```
rosbag play <rosbag>
```
