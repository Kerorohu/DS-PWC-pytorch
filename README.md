Official version(Caffe & PyTorch) is at https://github.com/NVlabs/PWC-Net, thank you all for attention.

# News
- Fix my usage of Correlation Layer, I've been using 19*19 neighborhood for matching.
    > NVIDIA is so kind to use their wonderful CUDA to let my mistake seem to be less stupid, btw I don't intend to remove my freaking slow Cost Volume Layer for code diversity or something.

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
    python3 main.py --input_norm --batch_norm --residual --corr Correlation --corr_activation pred --load example/SintelFinal-200K-noBN_SintelFinal-148K-BN.pkl -i example/1.png example/2.png -o example/output.flo
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
If there is any difference between your implementation and mine, please create an issue or something.
- Network Parameters
    ```
    Parameters: 8623340 Size: 32.89543151855469 MB
    ```
- Training Logs
    ```
    Step [100/800000], Loss: 0.3301, EPE: 42.0071, Forward: 34.287192821502686 ms, Backward: 181.38124704360962 ms
    Step [200/800000], Loss: 0.2359, EPE: 28.7398, Forward: 32.04517364501953 ms, Backward: 182.32821941375732 ms
    Step [300/800000], Loss: 0.2009, EPE: 24.3589, Forward: 31.214130719502766 ms, Backward: 182.9234480857849 ms
    Step [400/800000], Loss: 0.1802, EPE: 21.8847, Forward: 31.183505654335022 ms, Backward: 183.74325275421143 ms
    Step [500/800000], Loss: 0.1674, EPE: 20.4151, Forward: 30.955915451049805 ms, Backward: 183.9722876548767 ms
    Step [600/800000], Loss: 0.1583, EPE: 19.3853, Forward: 30.943967501322426 ms, Backward: 184.35366868972778 ms
    Step [700/800000], Loss: 0.1519, EPE: 18.6664, Forward: 30.953510829380583 ms, Backward: 184.56024714878626 ms
    Step [800/800000], Loss: 0.1462, EPE: 18.0256, Forward: 30.91249644756317 ms, Backward: 184.76592779159546 ms
    ```

