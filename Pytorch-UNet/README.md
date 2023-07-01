# U-Net: Semantic segmentation with PyTorch

## Quick start

### Without Docker

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch 1.13 or later](https://pytorch.org/get-started/locally/)

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download the data and run training:
```bash
bash scripts/download_data.sh
python train.py --amp
```

### With Docker

1. [Install Docker 19.03 or later:](https://docs.docker.com/get-docker/)
```bash
curl https://get.docker.com | sh && sudo systemctl --now enable docker
```
2. [Install the NVIDIA container toolkit:](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```
3. [Download and run the image:](https://hub.docker.com/repository/docker/milesial/unet)
```bash
sudo docker run --rm --shm-size=8g --ulimit memlock=-1 --gpus all -it milesial/unet
```

4. Download the data and run training:
```bash
bash scripts/download_data.sh
python train.py --amp
```


## Usage
**Note : Use Python 3.6 or newer**

### Docker

A docker image containing the code and the dependencies is available on [DockerHub](https://hub.docker.com/repository/docker/milesial/unet).
You can download and jump in the container with ([docker >=19.03](https://docs.docker.com/get-docker/)):

```console
docker run -it --rm --shm-size=8g --ulimit memlock=-1 --gpus all milesial/unet
```


### Training and evaluation:

Train model 1-channel:  
python3 train.py --epochs 100 --amp --classes 2 --batch-size 2 --channels 1 --scale 1

Train model 2-channel:  
python3 train.py --epochs 100 --amp --classes 2 --batch-size 2 --channels 2 --scale 1

Train model 3-channel:  
python3 train.py --epochs 100 --amp --classes 2 --batch-size 2 --channels 3 --scale 1

Train model 4-channel:  
python3 train.py --epochs 100 --amp --classes 2 --batch-size 2 --channels 4 --scale 1

(--load checkpoints4/checkpoint_epoch9.pth)

Predict mask:  
python3 predict.py -m checkpoints/checkpoint_epoch10.pth -i data/imgs/0055_0.png --viz --no-save

Evaluation 1-channel:  
python3 test.py --model checkpoints1/earlystop.pth --input data/validate/l/ --classes 2 --scale 1 --channels 1
python3 test.py --model checkpoints1/earlystop.pth --input data/test/l/ --classes 2 --scale 1 --channels 1


Evaluation 2-channel:  
python3 test.py --model checkpoints2/earlystop.pth --input data/validate/le/ --classes 2 --scale 1 --channels 2
python3 test.py --model checkpoints2/earlystop.pth --input data/test/le/ --classes 2 --scale 1 --channels 2


Evaluation 3-channel:  
python3 test.py --model checkpoints3/earlystop.pth --input data/validate/rgb/ --classes 2 --scale 1 --channels 3
python3 test.py --model checkpoints3/earlystop.pth --input data/test/rgb/ --classes 2 --scale 1 --channels 3


Evaluation 4-channel:  
python3 test.py --model checkpoints4/earlystop.pth --input data/validate/rgbe/ --classes 2 --scale 1 --channels 4
python3 test.py --model checkpoints4/earlystop.pth --input data/test/rgbe/ --classes 2 --scale 1 --channels 4

(--no-save for not saving output images)


Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox:

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)
