# Use an official Python 2.7 image as a base
FROM python:2.7

# Set the working directory inside the container
WORKDIR /app

# Install necessary packages to build CMake from source
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    libssl-dev

# Download, extract, and install CMake 3.15 or higher
RUN wget https://github.com/Kitware/CMake/releases/download/v3.15.7/cmake-3.15.7.tar.gz \
    && tar -xzvf cmake-3.15.7.tar.gz \
    && cd cmake-3.15.7 \
    && ./bootstrap \
    && make \
    && make install \
    && cd .. \
    && rm -rf cmake-3.15.7 cmake-3.15.7.tar.gz

# Install numpy
RUN pip install numpy

# Install geos
RUN wget https://download.osgeo.org/geos/geos-3.13.0.tar.bz2 \
    && tar xvfj geos-3.13.0.tar.bz2 \
    && cd geos-3.13.0 \
    && mkdir _build \
    && cd _build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local .. \
    && make \
    && ctest \
    && make install

# Install shapely
RUN wget https://pypi.python.org/packages/source/S/Shapely/Shapely-1.2.1.tar.gz#md5=da54c772443bd0398aa588e0f3e9c190 && pip install Shapely-1.2.1.tar.gz
#RUN pip install shapely

# Install descartes
RUN wget https://pypi.python.org/packages/source/d/descartes/descartes-1.0.tar.gz && pip install descartes-1.0.tar.gz

# Install the 'ipkiss' package
RUN git clone https://github.com/jtambasco/ipkiss.git && cd ipkiss && pip install .

# You can add any additional setup commands here
# For example: environment variables, or specific configurations

# Specify the entrypoint to make it easier to run scripts
ENTRYPOINT ["python"]
