# ZeroFlicks (VinesVideo)

This is the offical implementation of the ZeroFlicks, a zero shot video generation pipeline.

## News

* [06/05/2023] We have released the code for the ZeroFlicks pipeline.

<!-- ## TODO

Code check and clean
Add more details to README
Add Arxiv link
Add support for pip install -->


## Setup


### 1. Clone this repository and enter:

``` shell
git clone --recursive https://github.com/inf-monkeys/vines-video.git
cd vines-video/
```
### 2. Install requirements using Python 3.10 and CUDA >= 11.6, using conda and pip:
``` shell
conda conda env create -f environment.yml
conda activate zeroflicks
pip install -r requirements.txt
```

You are all set if you only want to use the controlled zero-shot video generation pipeline. 
### 3. [Optional] Environment setup for the two-stage zero-shot video generation pipeline

If you want to try our two-stage Zero-shot Text to Video Generation pipeline, you need to install a text to motion model. Our model can work with any off-the-shelf text to motion model, but we recommend using [MotionDiffusionModel](https://github.com/GuyTevet/motion-diffusion-model) for its simplicity and performance.



3. Install [MotionDiffusionModel](https://github.com/GuyTevet/motion-diffusion-model) following their instructions. Note that this repo is dependent on python3.7, but we have tested it with python3.10 and it works fine.

4. Put the MotionDiffusionModel repo under the same directory as this repo, and rename it to `motion-diffusion-model`.

<!-- ### Import from pip -->
TBA
<!-- We have also released a pip package for the zero-shot video generation pipeline. You can install it using:
``` shell
pip install zero-flicks
``` -->


## Zero-shot Text to Video Generation



## Controlled Zero-shot Text to Video Generation

## Acknowledgements

Our code is based on [Text2Video-Zero](https://github.com/Picsart-AI-Research/Text2Video-Zero/tree/main), please checkout their repo for more details.

Our two-stage zero-shot video generation pipeline is dependent on [MotionDiffusionModel](https://github.com/GuyTevet/motion-diffusion-model)

We thank Inf-Monkeys for sponsoring this research, please check out their [website](https://frame.infmonkeys.com/) for more awesome applications.