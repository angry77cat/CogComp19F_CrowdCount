Crowd Counting in Drone View
======

>This is the final project from Cognitive Computing 2019 Fall, we have implemented a crowd counting model by PyTorch. 

## Usage
To start with this project, please get the dataset first.

### Dataset
> dataset candidates:
> [Stanford Drone Dataset](http://cvgl.stanford.edu/projects/uav_data/) - Aerial Images over Standford Campus
> [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) - Object Detection in Images
> [UCF-QNRF](https://www.crcv.ucf.edu/data/ucf-qnrf/) - A Large Crowd Counting Data Set
> [ShanghaiTech_Crowd_Counting_Dataset](https://github.com/desenzhou/ShanghaiTechDataset)
> [GCC dataset](https://gjy3035.github.io/GCC-CL/)

#### Our Raw dataset schema

        / 
        |--raw_dataset/
        |    |--ShanghaiTech_Crowd_Counting_Dataset/
        |    |    |--part_A_final/
        |    |    |    |--test_data/
        |    |    |    |--train_data/
        |    |    |--part_B_final/
        |    |    |    |--test_data/
        |    |    |    |--train_data/
        |    |--Standford_Drone_Dataset/
        |    |    |--annotations/
        |    |    |--videos/
        |    |--UCF-QNRF_ECCV18/
        |    |    |--Test/
        |    |    |--Train/
        |    |--VisDrone2019/
        |    |    |--VisDrone2019-DET-Train/
        |    |    |    |--annotations/
        |    |    |    |--images/
        |    |    |--VisDrone2019-DET-val/
        |    |    |    |--annotations/
        |    |    |    |--images/

The unzipped data should be placed under `raw_dataset/` and we are going to extract frames from original movies and annotations to `data/`.  
 
The preprocessing details are shown in `drone_data_generator.ipynb`. In short, we only extract human-related labels such as pedestrians and biker. Finally, the input and output are both 2-dimensional images while the output is a 8-fold downsampled hot-map indicating the crowd density(hot-map). 

#### Our preprocessed data

        /
        |--data/
        |    |--StandfordDD_train/
        |    |--StandfordDD_val
        |    |--test
        |    |--VisDrone2019_train
        |    |--VisDrone2019_val


The preprocessed dataset is also provided [here](https://drive.google.com/drive/u/1/folders/1EsaYItpd2JU48udURYVIMkXHQh3Cf8B8). Or you can download it via `download_dataset.sh`.

    . download_dataset.sh $1 $2
 - `$1` is the fileid.  
 - `$2` is the filename.  

1. Standford drone dataset [23GB]
2877 for training, 20 for validation. Total: 68GB

        . download_dataset.sh 1AHkQAeYZcQm2h1mUoYwivt1kADp238Cj sdd.zip
    
2. Visdrone dataset [24GB]
3546 for training, 449 for validation. Total: 62.4GB

        . download_dataset.sh 1eXmn9qxzG5FabUqNexXxtGu_7jNbcjhO visdrone.zip
        
3. testing set [0.5GB]

        . download_dataset.sh 1HoAuI7Q6W7d9Rphp7RL-LlXqOy0-7RhH testset.zip

Example:
 
![](https://i.imgur.com/K0Occto.png)

 
For our final goal, we expect to do crowd counting for recent parades. We acknownledge a [Youtube channel](https://www.youtube.com/channel/UCJ_jxg20BXXDv-Z62rT7vyQ/videos) for providing high resolution footages in a drone view.

![](https://i.imgur.com/ZDrcpEA.jpg)


### Model Design
The main reference is from a repository [CSRNet-pytorch](https://github.com/leeyeehoo/CSRNet-pytorch).


### Download Pretrained Models
> under construction

### Result
> under construction, only preliminary is shown.
![](https://i.imgur.com/7Mo2bfT.png)

## References
1. [CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes](https://arxiv.org/pdf/1802.10062.pdf)
2. [Crowd counting via scale-adaptive convolutional neural network](https://arxiv.org/pdf/1711.04433.pdf)

## Packages
Below is a list of packages we used to implement this project:

> [`CUDA`](https://www.h5py.org/https://developer.nvidia.com/cuda-90-download-archive?target_os=Windows&target_arch=x86_64): 9.0  
> [`python`](https://www.python.org/): 3.7.3  
> [`torch`](https://pytorch.org/): 1.0.1  
> [`numpy`](http://www.numpy.org/): 1.16.2  
> [`pandas`](https://pandas.pydata.org/): 0.24.2  
> [`PIL`](https://pypi.org/project/Pillow/): 5.4.1  
> [`torchvision`](https://pypi.org/project/torchvision/): 0.2.2  
> [`cv2`](https://pypi.org/project/opencv-python/), [`matplotlib`](https://matplotlib.org/)   
> [The Python Standard Library](https://docs.python.org/3/library/)
