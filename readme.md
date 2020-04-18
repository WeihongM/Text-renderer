
### Synthetic_Textline_Data_For_Text_Recognition

---



#### 0.Install

```shell
# download anaconda and install(test on python3.5)

# install opencv
conda install -c menpo opencv3=3.1.0
# install Pygame
conda install -c cogsci pygame
# install lmdb
conda install -c conda-forge python-lmdb
# install other required package
conda install pandas, h5py, scipy, Pillow
```



#### 1.Run

```shell
python generate_training_data.py
```


#### 2. Code explanation

**main process:**

**(refer to paper: NIPS2014 Synthetic Data and Artificial Neural Networks for
Natural Scene Text Recognition)**[1]

1. Font redering
2. Border/shadow rendering
3. Base coloring
4. Projective distortion
5. Natural data blending
6. Noise


#### 3.Issues

> ImportError: libSDL-1.2.so.0: cannot open shared object file: No such file or directory

run command: sudo apt-get install libsdl1.2-dev

#### 4.Reference

[1] [Max Jaderberg's Text Renderer](https://arxiv.org/pdf/1406.2227.pdf)

[2] [Synthetic_Data_Engine_For_Text_Recognition](https://github.com/rayush7/Synthetic_Data_Engine_For_Text_Recognition)