# oni
Unofficial and imperfect implementation of ONI (Controllable Orthogonalization in Training DNNs. Lei Huang, Li Liu, Fan Zhu, Diwen Wan, Zehuan Yuan, Bo Li, Ling Shao. CVPR 2020)

I strongly recommend going to [the official repositry](https://github.com/huangleiBuaa/ONI).

This repositry is for an assignment of a lecture [("Visual Media" at the Univ of Tokyo)](https://www.hal.t.u-tokyo.ac.jp/~yamasaki/lecture/index.html).

# Training
For a 6-layer MLP on Fashion-MNIST with learning rate 0.05, run this.
```
$ python train.py --lr 0.05 --oni_itr 5 --depth 6 --batch_size 256 --dataset fmnist
```
`--oni_itr` specifies the iteration number of ONI. `--oni_itr 0` means OrthInit unless `--no_orthinit` is true.  

If `--no_scaling` is true, the scaling operation is not performed.

For a VGG-Style network on CIFAR-10 with g=4 and k=2, run this.
```
$ python train.py --lr 0.02  --epochs 160 --batch_size 128 --dataset cifar10 --g 4 --k 2
```
