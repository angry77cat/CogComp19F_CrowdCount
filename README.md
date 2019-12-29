Crowd Counting in Drone View
======

>This is the final project from Cognitive Computing 2019 Fall, we have implemented a crowd counting model by PyTorch. 

# Usage:
To start with this project, please get the dataset first.

### Dataset
We acknownledge [Stanford Drone Dataset](http://cvgl.stanford.edu/projects/uav_data/) for annotated movies. The unzipped data should be placed under `Standford_Drone_Dataset/` since we are going to extract frames from original movies and annotations to `data/raw`.  
 
The details are shown in `drone_data_generator.ipynb`. In short, the input and output are 2-dimensional images, the output is a 8-fold downsampled hot-map indicating the crowd density. Finally, It is expected to generate ~6k training data, each ~10MB. 
> The preprocessed data will be released soon, so that there's no need to preprocess again.
 
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
