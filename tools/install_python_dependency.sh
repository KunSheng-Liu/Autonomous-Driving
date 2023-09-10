# Install Tensorflow
sudo apt-get install -y \
    python3 \
    python3-pip

sudo pip3 install -U pip testresources setuptools==65.5.0
sudo pip3 install -U \
    numpy==1.21.1 \
    future==0.18.2 \
    mock==3.0.5 \
    keras_preprocessing==1.1.2 \
    keras_applications==1.0.8 \
    gast==0.4.0 \
    protobuf \
    pybind11 \
    cython \
    pkgconfig \
    packaging \
    h5py==3.6.0

sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v502 tensorflow==2.10.0+nv22.11

sudo pip3 install keras==2.10.0

# Install torch and torchvision
sudo apt-get -y update
sudo apt-get -y install \
    autoconf \
    bc \
    build-essential \
    g++-8 \
    gcc-8 \
    clang-8 \
    lld-8 \
    gettext-base \
    gfortran-8 \
    iputils-ping \
    libbz2-dev \
    libc++-dev \
    libcgal-dev \
    libffi-dev \
    libfreetype6-dev \
    libhdf5-dev \
    libjpeg-dev \
    liblzma-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libpng-dev \
    libreadline-dev \
    libssl-dev \
    libsqlite3-dev \
    libxml2-dev \
    libxslt-dev \
    locales \
    moreutils \
    openssl \
    python-openssl \
    rsync \
    scons \
    libopenblas-dev

sudo python3 -m pip install \
    aiohttp \
    scipy=='1.5.3' \
    protobuf

export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v502/pytorch/torch-1.13.0a0+936e9305.nv22.11-cp38-cp38-linux_aarch64.whl

export "LD_LIBRARY_PATH=/usr/lib/llvm-8/lib:$LD_LIBRARY_PATH"

sudo python3 -m pip install --no-cache $TORCH_INSTALL
sudo pip3 install torchvision

sudo apt-get install python3-opencv

# Install onnxruntime dependency
sudo apt-get update
sudo apt-get install -y \
	build-essential \
	curl \
	libcurl4-openssl-dev \
	libssl-dev \
	wget \
	python3 \
	python3-pip \
	git \
	tar
sudo pip3 install --upgrade pip
sudo pip3 install --upgrade setuptools
sudo pip3 install --upgrade wheel
sudo pip3 install numpy
sudo pip3 install packaging

sudo apt-get install -y --no-install-recommends \
	software-properties-common \
	libopenblas-dev \
	libpython3.8-dev \
	python3-dev

sudo apt-get install -y \
	protobuf-compiler \
	libprotoc-dev
