FROM osrf/ros:noetic-desktop-full

ENV PROJECT_DIR=/slam 

ARG ALGORITHM_FOLDER=rovtio

WORKDIR $PROJECT_DIR

RUN apt-get update && \
    apt install -y \
    git \
    python3-catkin-tools 

COPY $ALGORITHM_FOLDER .

#RUN git submodule update --init --recursive
RUN  sed -i "s/CV_GRAY2RGB/cv::COLOR_GRAY2RGB/" ./src/rovio/include/rovio/ImgUpdate.hpp  && \
     sed -i "s/CV_GRAY2RGB/cv::COLOR_GRAY2RGB/" ./src/rovtio/include/rovtio/ImgUpdate.hpp 

RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make_isolated -C ."
 
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash &&  catkin build rovio --cmake-args -DCMAKE_BUILD_TYPE=Release && \
     catkin build rovtio --cmake-args -DCMAKE_BUILD_TYPE=Release -DMAKE_SCENE=OFF -DROVIO_NCAM=2 -DROVIO_NMAXFEATURE=25"

RUN chmod +x launch.sh

ENTRYPOINT ["bash", "launch.sh"]
