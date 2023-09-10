# Autonomous Driving

## Prerequisites 
- Python packages
    1. Tensorflow ([Docs](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html#install))
        ```bash
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

        ```

    2. Pytorch ([Docs](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html))
        ```bash
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
        ```

        ```bash
        export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v502/pytorch/torch-1.13.0a0+936e9305.nv22.11-cp38-cp38-linux_aarch64.whl

        export "LD_LIBRARY_PATH=/usr/lib/llvm-8/lib:$LD_LIBRARY_PATH"

        sudo python3 -m pip install --no-cache $TORCH_INSTALL

        ```

    3. Torchvision ([Github](https://github.com/pytorch/vision/tree/main))
        ```bash
        git clone -b v0.14.0 https://github.com/pytorch/vision Torchvision
        ```
        ```bash
        cd Torchvision
        sudo python3 setup.py install
        sudo pip3 install torchvision==0.14.0
        ```
    3. Opencv for python
        ```bash
        sudo apt-get install python3-opencv
        ```


- OnnxRuntime library for c++
    1. Install dependencies ([Docs](https://onnxruntime.ai/docs/build/inferencing.html))
        
        ```bash
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
        ```
        
    2. Install [cmake-3.24](https://cmake.org/download/) or higher
        
        ```bash
        python3 -m pip install cmake
        cmake --version
        ```
        
    3. Download source code
        
        ```bash
        git clone https://github.com/microsoft/onnxruntime.git
        git submodule update --init --recursive --progress
        ```
        
    4. Compile the source code for Jetson device ([Docs](https://onnxruntime.ai/docs/build/eps.html#nvidia-jetson-tx1tx2nanoxavier))
        
        ```bash
        cd onnxruntime
        
        export PATH=/usr/local/cuda/bin:$(PATH)
        export CUDA_PATH=/usr/local/cuda
        export cuDNN_PATH=/usr/lib/aarch64-linux-gnu
        export CMAKE_ARGS="-DONNX_CUSTOM_PROTOC_EXECUTABLE=/usr/bin/protoc"
        
        ./build.sh --config Release --update --enable_pybind --build_shared_lib --build --parallel --build_wheel \
         --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu
        ```
        
        - `--build_shared_lib`: build the c++ library
        - `--parallel`: build in multi-thread
    5. List results
        
        ```bash
        ls -l build/Linux/Release/*.so
        ls -l build/Linux/Release/dist/*.whl
        ```
        
    6. Build onnxruntime for python
        
        ```bash
        pip3 install build/Linux/Release/dist/*.whl
        ```
        
    7. Build c/c++ shared library
        
        ```bash
        cd build/Linux/Release/
        make && sudo make install
        ```
    8. Fix include path

        ```bash
            cd /usr/local/include/onnxruntime
            sudo mv core/* ./
            sudo rmdir core
        ```
    9. Sync environment

        ```bash
            sudo ldconfig
            sudo ldconfig -p | grep onnx
        ```

- Opencv
    1. Install by script

        ```bash
            cd tools/
            chmod 777 install_opencv_4.6.0_Jetson.sh
            sudo ./install_opencv_4.6.0_Jetson.sh
        ```
    2. Fix include path

        ```bash
            cd /usr/local/include
            sudo mv opencv4/opencv2 ./
            sudo rmdir opencv4
        ```
    3. Sync environment

        ```bash
            sudo ldconfig
            sudo ldconfig -p | grep opencv
        ```


## Dataset
- Waymo dataset ([Docs](https://waymo.com/open/)) 
    1. Totorial ([Git](https://github.com/waymo-research/waymo-open-dataset))
    2. Data ([Cloud](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_4_1;tab=objects?pli=1&prefix=&forceOnObjectsSortingFiltering=false))


- Prepocess dataset
    ```bash
    parser_raw_dataset.py
    ```
    note: should modify the src/dist path

## Onnx Model
- Create model
    ```bash
    onnx_resnet50.ipynb
    ```

## Run the code
- Compile
    ```bash
    make debug RT_CPS
    ```

- Run
    ```bash
    gdb ./RT_CPS
    ```
