# Towards Long Horizon Action through Prompting Pre-trained Agent in Minecraft

#### Authors: Yuyang Tang (yuyang.tang@mail.utoronto.ca), Shijia Liu (shijia.liu@mail.utoronto.ca), Zexin Xue (zexin.xue@mail.utoronto.ca)

## Our paper report is available!! Check `GPTsteve1.pdf` for the full draft. 

## Description

In this paper, we expand on the previous research on autonomous agents in Minecraft, specifically STEVE-1. We present a new approach that leverages the reasoning capability of LLM to guide STEVE-1 agents through instructions, allowing them to accomplish long-horizon and multi-step tasks. We view the LLMs as high-level policies, which will implicitly create an action trajectory to complete the specified long-term task. We apply a benchmark with three long-horizon tasks for the agent to complete. Afterward, we analyze the gameplay recordings of the agents and assess the feasibility of our approach. Finally, we analyze the effectiveness of our method and suggest areas for improvement in future studies.

<!-- Our Structure wil be more complicated, but this will do as a placeholder -->

## Running our Code

### Setup

We provided our setup procedure in a shell script (`env_setup_log.sh`). 
**The script is a guide, not an executable!** This means you might want to run each step independently.  
All dependencies are tested to work on Ubuntu 22.04.

#### Notes:

> There might be some warnings from pip on conflicting `gym` version, feel free to ignore that.

> Some extra effort is needed to setup the `JAVA_HOME` environment variable. The same location has to be available in your `PATH` as well.

> To run python script on a headless machine, follow the template:

```bash
xvfb-run python path/to/minedojo/python/scripts.py

MINEDOJO_HEADLESS=1 python path/to/minedojo/python/scripts.py
```

> Please install MineDojo from the source (included in this repo), the reason being Malmo has an unsupported link that we fixed by hosting the file locally; otherwise, MineDojo will not run.

### Docker

An experimental docker environment is also available with all the dependencies installed. Use `docker pull barry121/ltlsteve:latest` to obtain the image.

You can run the image on your local machine or a headless machine with GPU support (requires nvidia docker toolkit, see [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)) with the command:

```bash
docker run --gpus all -it -d -p 8080:8080 barry0121/ltlsteve:latest tail -f /dev/null
docker exec -it <running_container_id> /bin/bash
```

### Run Experiments

First, please make sure you download the pre-trained weights by running `./STEVE/download_weight.sh`. Then you can choose one of the three options below: 

**Option 1** Run a single recorded trial

To record a single trial just run the following with target objective: 

```bash
cd STEVE
python steve1/run_agent/run_agent.py \
    --in_model data/weights/vpt/2x.model \
    --in_weights data/weights/steve1/steve1.weights \
    --prior_weights data/weights/steve1/steve1_prior.pt \
    --text_cond_scale 6.0 \
    --visual_cond_scale 7.0 \
    --gameplay_length $3 \
    --save_dirpath $test_dir \
    --task $1 \
    --start_template_pth $(pwd)/steve1/data/prompt/start_prompt.json \
    --next_template_pth $(pwd)/steve1/data/prompt/next_prompt.json \
    --help_template_pth $(pwd)/steve1/data/prompt/help_prompt.json \
    --stuckpoint_pth $(pwd)/steve1/data/prompt \
    --help_gap 2000 
```

**Option 2** Run a couple of recorded trials

We provided a bash script, and you can run it as follows: 

```bash
cd STEVE
./run_agent/run_test.sh "[task name]" [number of trials] [length of each trial] 
```
> The length of each trial is specified with a number representing how many frames the episode will last. We are limiting our environment to run at 30 FPS, so to generate a one-minute-long clip is equivalent to 1800 frames. 

The recorded video and communication log will be stored in `STEVE/data/[task name]`. 

**Option 3** Run STEVE-1 interactively (Not available on a headless machine) 

Execute the following command, and follow the instructions on the interactive GUI. 

```bash
python steve1/run_agent/run_interactive.py \
  --in_model data/weights/vpt/2x.model \
  --in_weights data/weights/steve1/steve1.weights \
  --prior_weights data/weights/steve1/steve1_prior.pt \
  --output_video_dirpath data/generated_videos/interactive_videos \
  --cond_scale 6.0
```
