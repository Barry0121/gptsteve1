# Towards Long Horizon Action through Prompting Pre-trained Agent in Minecraft

#### Authors: Yuyang Tang, Shijia Liu, Barry Xue

## Description

[insert description]

<!-- Our Structure wil be more complicated, but this will do as a placeholder -->

## Running our Code

### Setup

We provided a setup script (`env_setup_log.sh`) to create a conda environment with all the dependencies installed. This script is tested on Ubuntu 22.04.
There might be some warnings from pip on conflicting `gym` version, feel free to ignore that.

If you are running into issue relate to installation, please refer to the reference source listed for each steps in the shell script, and
install all dependencies individually.

> Some extra effort is needed to setup the `JAVA_HOME` environment variable. The same location has to be available in your `PATH` as well.

> To run python script on a headless machine, follow the template:

```bash
xvfb-run python path/to/minedojo/python/scripts.py

MINEDOJO_HEADLESS=1 python path/to/minedojo/python/scripts.py
```

### Supported Actions

[insert instructions]

## Results

[insert table and graphs]

## Acknowledgement

[insert all used resources]
