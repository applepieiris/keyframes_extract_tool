Created on Tue Dec  4 16:48:57 2018

keyframes extract tool

this key frame extract algorithm is based on interframe difference.

The principle is very simple
First, we load the video and compute the interframe difference between each frames

Then, we can choose one of these three methods to extract keyframes, which are 
all based on the difference method:
    
1. use the difference order
    The first few frames with the largest average interframe difference 
    are considered to be key frames.
2. use the difference threshold
    The frames which the average interframe difference are large than the 
    threshold are considered to be key frames.
3. use local maximum
    The frames which the average interframe difference are local maximum are 
    considered to be key frames.
    It should be noted that smoothing the average difference value before 
    calculating the local maximum can effectively remove noise to avoid 
    repeated extraction of frames of similar scenes.

After a few experiment, the third method has a better key frame extraction effect.

The original code comes from the link below, I optimized the code to reduce 
unnecessary memory consumption.
https://blog.csdn.net/qq_21997625/article/details/81285096

