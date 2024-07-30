
#OS setup
apt-get update && apt-get install nano unzip ffmpeg libsm6 libxext6 unar vim htop unzip gcc curl g++ python3-distutils python3-apt pip -y

#install python requirements
pip install -r requirements.txt

#copy over blender assets
scp -r justin@131.215.79.142:/home/justin/repos/animal-pointodyssey/data .
