# Towards Long Horizon Action through Prompting Pre-trained Agent in Minecraft

#### Authors: Yuyang Tang, Shijia Liu, Barry Xue

## Description

[insert description]

<!-- Our Structure wil be more complicated, but this will do as a placeholder -->

## Running our Code

### Setup

We provided a setup script (`env_setup_log.sh`) to create a conda environment with all the dependencies installed. This script is tested on Ubuntu 22.04.
There might be some warnings from pip on conflicting `gym` version, feel free to ignore that.

If you are running into issue relate to installation, please refer to the reference source listed for each steps in the shell script, and install all dependencies individually.

#### Notes:

> Some extra effort is needed to setup the `JAVA_HOME` environment variable. The same location has to be available in your `PATH` as well.

> To run python script on a headless machine, follow the template:

```bash
xvfb-run python path/to/minedojo/python/scripts.py

MINEDOJO_HEADLESS=1 python path/to/minedojo/python/scripts.py
```

> Please install MineDojo from source (include in this repo), reason being Malmo has unsupported link that we fixed by hosting the file locally; otherwise, MineDojo will not run.

### Docker

An experimental docker environment is also avaliable with all the dependencies installed. Use `docker pull barry121/ltlsteve:latest` to obtain the image.

You can run the image on your local machine or a headless machine with GPU support (requires nvidia docker toolkit, see [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)) with the command:

```bash
docker run --gpus all -it -d -p 8080:8080 barry0121/ltlsteve:latest tail -f /dev/null
docker exec -it <running_container_id> /bin/bash
```

### Run Experiments

[insert instructions]

## Results

[insert table and graphs]

## Acknowledgement

[insert all used resources]
