# Run this shell script at the root directory (/ltl_steve1/) to setup the environment

# Create Environment at local directory
conda create --prefix=ltlsteve python=3.10
activate ./ltlsteve

# Install PyTorch with cuda support (change the cuda version accordingly)
# Note: if using slurm, please initialize the repo inside of a gpunode and then install
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install Java 8 (reference: https://docs.minedojo.org/sections/getting_started/install.html) mine is for ubuntu
# requires root access
sudo apt update -y
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:openjdk-r/ppa
sudo apt update -y
sudo apt install -y openjdk-8-jdk
sudo update-alternatives --config java # run this to switch version

# Install headless renderer xvfb and denpendencies
sudo apt install xvfb xserver-xephyr tigervnc-standalone-server python3-opengl ffmpeg

# Prepare for gym installation (reference: https://stackoverflow.com/questions/77124879/pip-extras-require-must-be-a-dictionary-whose-values-are-strings-or-lists-of)
# pip install setuptools==65.5.0 pip==21
# pip install wheel==0.38.0

# Install MineDojo and MineCLIP
# (for debugging) Change the `build.gradle` dependency's file path, the comments should point out which line to change.
# (reference: https://github.com/MineDojo/MineDojo/issues/113)
cd MineDojo
pip install -e .
cd ..

# Install MineRL (This will take a while)
pip install git+https://github.com/minerllabs/minerl@v1.0.1

# Install VPT dependencies (VPT will downgrade gym to an older version and this is okay)
# pip install gym==0.19 gym3 attrs opencv-python # TODO: Might have to try changing VPT's dependency from 0.19 to 0.21, otherwise MineDojo can't run
pip install gym3 attrs opencv-python

# Extra dependencies
pip install gdown tqdm accelerate==0.18.0 wandb importlib-resources==5.0

# Clone Steve-1 and install locally
cd STEVE
pip install -e .
cd ..

# (Optional) If there is an issue when running test_env.py (ex: error related to Malmo), try to run the following:
# sudo apt-get install x11-xserver-utils