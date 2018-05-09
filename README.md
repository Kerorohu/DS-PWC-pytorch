# Acknowledgments
- [NVIDIA/flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch): framework, data transformers, loss functions, and many details about flow estimation.
- [yunjey/pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard): Tensorboard logger
- [sksq96/pytorch-summary](https://github.com/sksq96/pytorch-summary): model summary similar to `model.summary()` in Keras

# PWC-Net
This is an unofficial pytorch implementation of CVPR2018 paper: Deqing Sun *et al.* **"PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume"**.    
**Resources**  [arXiv](https://arxiv.org/abs/1709.02371) | [Caffe](https://github.com/deqings/PWC-Net)(official)

![](https://github.com/nameless-Chatoyant/PWC-Net_pytorch/raw/master/example/flow.png)
<p align="center">(flow outputs from top to bottom, the rightest is groundtruth)</p>

It starts to output reasonable flows. However, both time and performance need to be improved. Hope you have fun with this code, and feel free to share your idea about network and its hyper parameters.


# Usage
- **Requirements**
    - Python 3.6+
    - **PyTorch 0.4.0**
    - Tensorflow


- **Get Started with Demo**    
    Note that we only save weights of parameters instead of entire network, provided model file is for default configs, we may upload more advanced models in the future.
    ```
    python3 main.py --input_norm --batch_norm --residual --corr_activation pred --load models/best.pkl -i example/1.png example/2.png -o example/output.flo
    ```

- **Prepare Datasets**
    - Download [FlyingChairs](https://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip) for training  
        filetree when setting `--dataset FlyingChairs --dataset_dir <DIR_NAME>`
        ```
        <DIR_NAME>
        ├── 00001_flow.flo
        ├── 00001_img1.ppm
        ├── 00001_img2.ppm
        ...
        ```
    - Download [FlyingThings](https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/derived_data/flyingthings3d__optical_flow.tar.bz2) for fine-tuning  
        filetree when setting `--dataset FlyingThings --dataset_dir <DIR_NAME>`
        ```
        <DIR_NAME>
        ```
    - Download [MPI-Sintel](http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip) for fine-tuning if you want to validate on MPI-Sintel  
        filetree when setting `--dataset Sintel --dataset_dir <DIR_NAME>`
        ```
        <DIR_NAME>
        ├── training
        |   ├── final
        |   ├── clean
        |   ├── flow
        |   ...
        ├── test
        ...
        ```
    - Download [KITTI](http://www.cvlibs.net/download.php?file=data_scene_flow.zip) for fine-tuning if you want to validate on KITTI  
        filetree when setting `--dataset KITTI --dataset_dir <DIR_NAME>`
        ```
        <DIR_NAME>
        ├── training
        |   ├── image_2
        |   ├── image_3
        |   ...
        └── testing
        ```

- **Install Correlation Package**
    If you want to use correlation layer (`--corr Correlation`), please follow [NVIDIA/flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch) to install extra packages.
- **Train**
    ```
    python3 main.py train --dataset <DATASET_NAME> --dataset_dir <DIR_NAME>
    ```


# Details
## Network Parameters
Parameters: 8623340 Size: 32.89543151855469 MB
