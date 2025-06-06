


#OS setup
apt-get update && apt-get install nano unzip ffmpeg libsm6 libxext6 unar vim htop unzip gcc curl g++ python3-distutils python3-apt pip -y

#install python requirements
pip install -r requirements.txt

wget https://mirrors.iu13.net/blender/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz
tar -xJf blender-4.2.0-linux-x64.tar.xz
mv blender-4.2.0-linux-x64 /usr/bin/
export PATH="$PATH:/usr/bin/blender-4.2.0-linux-x64"
echo export PATH="\$PATH:/path/to/dir" >> ~/.bashrc

#copy over blender assets
scp -r justin@131.215.79.142:/home/justin/repos/animal-pointodyssey/data .
