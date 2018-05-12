# Self Driving Drone

Tensorflow implementation of self driving drone using Deep Double Dueling Q Network and Galaga method.

Microsoft AirSim is needed.

For random target, it achieves about 70% success rate using 34 layer ResNET and 3 layer fully connected.

It uses depth map for image processing, and I think deeper network, like 50 layer ResNET, will be needed to use raw RGB input.

[Training Environment]

- CPU: AMD Ryzen 7 1700
- GPU: nvidia GTX1080Ti GDDR5X 11GB (Factory Overclocked)
- Training Time: under 8 hours
