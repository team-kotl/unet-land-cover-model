### :skull_emoji: XD

> Use WSL!

BigEarth Dataset

`https://bigearth.net/#downloads`

Corine Dataset

`https://land.copernicus.eu/en/products/corine-land-cover/clc2018#download`

or

`https://drive.google.com/drive/folders/1pCc9sbzemzAgFc_2F08ObK5p0Zvm6pug?usp=sharing`

Do this first before doing anything in order to preserve your personal computer.

```bash
conda create -n landcover-unet
conda activate landcover-unet
conda install -c conda-forge --file conda-requirements.txt
pip install -r requirements.txt
```