# cv_morph
reimplement of Morph function in OpenCV (https://github.com/Itseez/opencv/blob/master/modules/imgproc/src/morph.cpp)

The morph function in OpenCV may make some faults when running on a virtual machine. I guess it is caused by the accelerated library of opencv. However, even I disable all accelerated library such as TBB, OpenMP, I still get the fault from the morph function. The fault is, when running an erode or a dilate function, the function executes weakly compared with the true result and it looks like the function didn't run at all!

So I have no way but rewrite the morph function. But now I haven't made any optimization. I would update this morph functions as I made some progress.
