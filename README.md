# ZeroFlicks (VinesVideo)

This is the offical implementation of the ZeroFlicks, a zero shot video generation pipeline.

## News

* [06/09/2023] We have uploaded the api for the two-stage zero-shot video generation pipeline.
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

Install [MotionDiffusionModel](https://github.com/GuyTevet/motion-diffusion-model). This should be already done if you have followed the previous step, the "--recursive" flag will automatically clone the MotionDiffusionModel repo. If you have not, you can do it manually by:
``` shell
git clone https://github.com/GuyTevet/motion-diffusion-model
cp -r motion-diffusion-model/ zeroflicks/
```
You also need to download their pretrained models following their instructions, and put them into the corresponding folders under `zeroflicks/motion-diffusion-model/`.


We have modified the plot function in motion-diffusion-model to support our two-stage pipeline. You need to overwrite the original plot function with our modified version:
``` shell
cp ./misc/plot_script.py ./motion-diffusion-model/data_loaders/humanml/utils/plot_script.py
```

That's it! You are all set to use our two-stage zero-shot video generation pipeline.


## Zero-shot Text to Video Generation
We prepared the `go_zero.py` to provide an example of how to use our zero-shot video generation api. You can run it with:
``` shell
python go_zero.py
```
The results will be under `output`.


## Controlled Zero-shot Text to Video Generation
TBA

## Controlled Zero-shot Text to Video Generation with dreambooth
TBA

# Acknowledgements

We thank Inf-Monkeys for sponsoring this research, please check out their [website](https://frame.infmonkeys.com/) for more awesome applications.

Our code is based on [Text2Video-Zero](https://github.com/Picsart-AI-Research/Text2Video-Zero/tree/main), please checkout their repo for more details.

Our two-stage zero-shot video generation pipeline is dependent on [MotionDiffusionModel](https://github.com/GuyTevet/motion-diffusion-model)
