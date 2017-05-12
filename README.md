# deep-rover

## Installation Instructions
Setup a virtual env using python 3.6

pip install gym keras tensorflow

If you are not in a virtual env (you really should be), make sure you are using the proper version of pip and python. On OS X, the default installation is python2.6. To avoid using it, use pip3 and python3.

For optimal performance, considering building tensorflow from source (https://www.tensorflow.org/install/install_sources)

## Installation Instructions (How to create a virtual env without root access)

In directory $DIR 

### 1) Create directories
mkdir .localpython

mkdir src && cd src

### 2) Install Python
wget http://www.python.org/ftp/python/3.6.1/Python-3.6.1.tgz

tar -xzvf Python-3.6.1 && cd Python-3.6.1

./configure --prefix=$DIR/.localpython

make

make install

### 3) Install Virtual Env
wget https://pypi.python.org/packages/d4/0c/9840c08189e030873387a73b90ada981885010dd9aea134d6de30cd24cb8/virtualenv-15.1.0.tar.gz#md5=44e19f4134906fe2d75124427dc9b716

tar -xzvf virtualenv-15.1.0.tar.gz

cd virtualenv-15.1.0

../../.localpython/bin/python3 setup.py install

### 4) Create Virtual Env
./python-dist/.localpython/bin/virtualenv --python=./python-dist/.localpython/bin/python3.6 rover-env
