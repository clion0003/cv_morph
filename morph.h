#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

bool morph_dilate(Mat &src, Mat &dst, int height, int width, int pointh, int pointw)
{
	src.copyTo(dst);
	if (height > src.rows || width > src.cols || height < 0 || width < 0 || pointh < 0 || pointw<0 || pointh>=height || pointw >= width)
	{
		return false;
	}
	for (int i = 0; i < src.rows-height; i++)
	{
		for (int j = 0; j < src.cols-width; j++)
		{
			uchar tmp = 0;
			for (int k = 0; k < height; k++)
			{
				for (int l = 0; l < width; l++)
				{
					tmp |= src.ptr<uchar>(i + k)[j + l];
				}

			}
			dst.ptr<uchar>(i + pointh)[j + pointw] = tmp;
		}
	}
	return true;
}

bool morph_erode(Mat &src, Mat &dst, int height, int width, int pointh, int pointw)
{
	src.copyTo(dst);
	if (height > src.rows || width > src.cols || height < 0 || width < 0 || pointh < 0 || pointw<0 || pointh>=height || pointw >= width)
	{
		return false;
	}
	for (int i = 0; i < src.rows - height; i++)
	{
		for (int j = 0; j < src.cols - width; j++)
		{
			uchar tmp = 255;
			for (int k = 0; k < height; k++)
			{
				for (int l = 0; l < width; l++)
				{
					tmp &= src.ptr<uchar>(i + k)[j + l];
				}

			}
			dst.ptr<uchar>(i + pointh)[j + pointw] = tmp;
		}
	}
	return true;
}
