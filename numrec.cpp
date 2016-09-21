
#define DLL_IMPLEMENT


#include "opencv/cv.h"
#include "numrec.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#define OPENCV_SHOW
//#define LINUX_DEBUG
#ifdef __linux__
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <sys/types.h>
#include <string.h>
#include <sys/sysinfo.h>
#endif


//#pragma comment( lib, "IlmImfd.lib")
//#pragma comment( lib, "libjasperd.lib")
//#pragma comment( lib, "libjpegd.lib")
//#pragma comment( lib, "libpngd.lib")
//#pragma comment( lib, "libtiffd.lib")
//#pragma comment( lib, "zlibd.lib")
//
//#pragma comment( lib, "vfw32.lib" )  
//#pragma comment( lib, "comctl32.lib" ) 

#define NUMREC_W_FRAC 0.215
#define NUMREC_H_FRAC 0.111
#define NUMREC_THRESHOLD 0.55

#define NUMBER_INSIDE_W_FRAC 0.25
#define NUMBER_INSIDE_H_FRAC 0.15
#define NUMBER_INSIDE_THRESHOLD 0.52

#define WHB_ONE_THRESHOLD 0.34
#define IS_ONE_THRESHOLD 0.4
#define ONE_MID_THRESHOLD 0.6
#define WHB_THRESHOLD_DOWN 0.34
#define WHB_THRESHOLD_UP 1.1

#define COUNT_THRESHOLD 5

#define HORIZ_UP 1
#define VERT_LEFT_UP 2
#define VERT_RIGHT_UP 4
#define HORIZ_MID 8
#define VERT_LEFT_DOWN 16
#define VERT_RIGHT_DOWN 32
#define HORIZ_DOWN 64
#define ALL_SEGS 127

#define D_ZERO (ALL_SEGS & ~HORIZ_MID)
#define D_ONE (VERT_RIGHT_UP | VERT_RIGHT_DOWN)
#define D_TWO (ALL_SEGS & ~(VERT_LEFT_UP | VERT_RIGHT_DOWN))
#define D_THREE (ALL_SEGS & ~(VERT_LEFT_UP | VERT_LEFT_DOWN))
#define D_FOUR (ALL_SEGS & ~(HORIZ_UP | VERT_LEFT_DOWN | HORIZ_DOWN))
#define D_FIVE (ALL_SEGS & ~(VERT_RIGHT_UP | VERT_LEFT_DOWN))
#define D_SIX (ALL_SEGS & ~VERT_RIGHT_UP)
#define D_SEVEN (HORIZ_UP | VERT_RIGHT_UP | VERT_RIGHT_DOWN)
#define D_ALTSEVEN (VERT_LEFT_UP | D_SEVEN)
#define D_EIGHT ALL_SEGS
#define D_NINE (ALL_SEGS & ~VERT_LEFT_DOWN)
#define D_ALTNINE (ALL_SEGS & ~(VERT_LEFT_DOWN | HORIZ_DOWN))

#define MAX_ITEM 40
#define THRESH_ONE  50

using namespace cv;
using namespace std;

struct numberRec{
	int number;
	int size_w;
	int size_h;
	int pos_x;
	int pos_y;
};

struct pressure_num{
	int value;
	int height;
};

double angle(CvPoint* pt1, CvPoint* pt2, CvPoint* pt0)
{
	double dx1 = pt1->x - pt0->x;
	double dy1 = pt1->y - pt0->y;
	double dx2 = pt2->x - pt0->x;
	double dy2 = pt2->y - pt0->y;
	return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// 返回图像中找到的所有轮廓序列，并且序列存储在内存存储器中
CvSeq* findSquares4(IplImage* img, CvMemStorage* storage, int &area)
{

	CvSeq* contours;
	int minArea = 9999999;
	int i, c, l, N = 11;
	CvSize sz = cvSize(img->width & -2, img->height & -2);
	//cout<<hex<<"yes"<<(long long)img<<endl;

	IplImage* timg = cvCloneImage(img);
	//IplImage* newimg = cvCreateImage(sz,8,3);
	IplImage* gray = cvCreateImage(sz, 8, 1);
	IplImage* pyr = cvCreateImage(cvSize(sz.width / 2, sz.height / 2), 8, 3);
	IplImage* tgray;
	CvSeq* result;
	double s, t;
	// 创建一个空序列用于存储轮廓角祦E
	CvSeq* squares = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvPoint), storage);
	cvSetImageROI(timg, cvRect(0, 0, sz.width, sz.height));
	// 过滤詠E丒
	cvPyrDown(timg, pyr, 7);
	cvPyrUp(pyr, timg, 7);
	tgray = cvCreateImage(sz, 8, 1);
	// 簛E汤?色分别尝试提取
	bool flag = false;
	for (c = 0; c < 3; c++)
	{
		// 提取 the c-th color plane
		cvSetImageCOI(timg, c + 1);
		cvCopy(timg, tgray, 0);

		// 尝试各种阈值提取得到的（N=11）
		for (l = 0; l < N; l++)
		{
			// apply Canny. Take the upper threshold from slider
			// Canny helps to catch squares with gradient shading  
			if (l == 0)
			{
				cvCanny(tgray, gray, 0, THRESH_ONE, 5);
				//使用任意结构元素膨胀图蟻E
				cvDilate(gray, gray, 0, 1);
			}
			else
			{
				// apply threshold if l!=0:
				cvThreshold(tgray, gray, (l + 1) * 255 / N, 255, CV_THRESH_BINARY);
			}

			// 找到所有轮廓并且存储在序列中
			cvFindContours(gray, storage, &contours, sizeof(CvContour),
				CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

			// 遍历找到的每个轮廓contours
			while (contours)
			{
				//用指定精度逼絹E啾咝吻?
				result = cvApproxPoly(contours, sizeof(CvContour), storage,
					CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0);


				if (result->total == 4 &&
					fabs(cvContourArea(result, CV_WHOLE_SEQ)) > 600000 &&
					fabs(cvContourArea(result, CV_WHOLE_SEQ)) < 1400000 &&
					cvCheckContourConvexity(result))
				{
					//Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
					//cvDrawContours(newimg, contours, color, color, -1, CV_FILLED, 8);
					s = 0;
					//std::cout << fabs(cvContourArea(result, CV_WHOLE_SEQ)) << std::endl;
					for (i = 0; i < 5; i++)
					{
						// find minimum angle between joint edges (maximum of cosine)
						if (i >= 2)
						{
							t = fabs(angle(
								(CvPoint*)cvGetSeqElem(result, i),
								(CvPoint*)cvGetSeqElem(result, i - 2),
								(CvPoint*)cvGetSeqElem(result, i - 1)));
							s = s > t ? s : t;
						}
					}

					// if 余弦值 足够小，可以认定角度为90度直角
					//cos0.1=83度，能较好的趋絹E苯?
					if (s < 0.15){
						if (fabs(cvContourArea(result, CV_WHOLE_SEQ)) < minArea){
							for (i = 0; i < 4; i++)
								cvSeqPush(squares,
								(CvPoint*)cvGetSeqElem(result, i));
							minArea = (int)fabs(cvContourArea(result, CV_WHOLE_SEQ));
							flag = true;
							//cout << "find a contour" << endl;
						}
					}
				}

				// 继续查找下一个轮廓
				contours = contours->h_next;
			}
		}
	}

	//cvSaveImage("src.jpg", newimg);
	cvReleaseImage(&gray);
	cvReleaseImage(&pyr);
	cvReleaseImage(&tgray);
	cvReleaseImage(&timg);

	if (!flag)return NULL;
	//cvReleaseImage(&newimg);
	else return squares;
}


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

int findItems(Mat src, vector<Rect> &items
#if defined(__linux__) && defined(LINUX_DEBUG)
	,const string image_tmp_path
#endif
	){
	RNG rng(12345);
	//imshow("out", src);
	Mat mid1,mid2,mid3,mid4,mid5;
	int i;
	for (i = 0; i < 60; i = i + 3){
		int count = 0;
		for (int j = 0; j < src.cols; j++)
		{
			if (src.ptr<uchar>(i)[j] == 0)count++;
			if (src.ptr<uchar>(i + 1)[j] == 0)count++;
			if (src.ptr<uchar>(i + 2)[j] == 0)count++;
		}
		if (count < src.cols*1.2)break;
	}
	src.copyTo(mid1);
	rectangle(mid1, Point(0, 0), Point(699, 799), 255, i + 5, 8, 0);

#if (defined(_WIN32)||defined(_WIN64)) && defined(OPENCV_SHOW)
	imshow("1", mid1);
#elif defined(__linux__) && defined(LINUX_DEBUG)
	string image_tmp;
	image_tmp = image_tmp_path + "/1.jpg";
	imwrite(image_tmp.c_str(), mid1);
#endif

	//morphologyEx(mid1, mid2, MORPH_DILATE, Mat(3, 3, CV_8U), Point(-1, -1), 1);
	morph_dilate(mid1, mid2, 3, 3, 1, 1);
#if (defined(_WIN32)||defined(_WIN64)) && defined(OPENCV_SHOW)
	imshow("2", mid2);
	//waitKey();

#elif defined(__linux__) && defined(LINUX_DEBUG)
	image_tmp = image_tmp_path + "/2.jpg";
	imwrite(image_tmp.c_str(), mid2);
#endif

	//morphologyEx(mid2, mid3, MORPH_ERODE, Mat(3, 3, CV_8U), Point(-1, -1), 1);
	morph_erode(mid2, mid3, 3, 3, 1, 1);
#if (defined(_WIN32)||defined(_WIN64)) && defined(OPENCV_SHOW)
	imshow("3", mid3);

#elif defined(__linux__) && defined(LINUX_DEBUG)
	image_tmp = image_tmp_path + "/3.jpg";
	imwrite(image_tmp.c_str(), mid3);
#endif

	//morphologyEx(mid3, mid4, MORPH_ERODE, Mat(11, 3, CV_8U), Point(-1, -1), 1);
	morph_erode(mid3, mid4, 11, 3, 5, 1);
#if (defined(_WIN32)||defined(_WIN64)) && defined(OPENCV_SHOW)
	imshow("4", mid4);

#elif defined(__linux__) && defined(LINUX_DEBUG)
	image_tmp = image_tmp_path + "/4.jpg";
	imwrite(image_tmp.c_str(), mid4);
#endif

	//morphologyEx(mid4, mid5, MORPH_DILATE, Mat(9, 3, CV_8U), Point(-1, -1), 1);
	morph_dilate(mid4, mid5, 9, 3, 4, 1);
#if (defined(_WIN32)||defined(_WIN64)) && defined(OPENCV_SHOW)
	imshow("5", mid5);
	waitKey();
#elif defined(__linux__) && defined(LINUX_DEBUG)
	image_tmp = image_tmp_path + "/5.jpg";
	imwrite(image_tmp.c_str(), mid5);
#endif
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	threshold(mid5, threshold_output, 100, 255, THRESH_BINARY);
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	//imshow("aaa", src);
	/// 多边形逼絹E掷?+ 获取矩形边界縼E
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
	}

	//Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	int count = 0;
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		//drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		//rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
		if (boundRect[i].width < 50 && boundRect[i].height < 50) continue;
		if (boundRect[i].width >300 || boundRect[i].height > 300) continue;
		items[count] = boundRect[i];
		count++;
		//rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), 255, 2, 8, 0);
	}
	//imshow("Contours", drawing);

	return 0;
}

bool Is_One(Mat Img){
	int w = Img.cols;
	int h = Img.rows;
	if ((double)w / (double)h > 0.5)return 0;
	int black_number = 0, tmp = 0, all = 0, white_number = 0, ex_number = 0;

	black_number = 0;
	for (int i = 0; i < w; i++){
		tmp = 0;
		for (int j = 0; j < h; j++)
		{
			if (Img.ptr<uchar>(j)[i] == 0)tmp++;
		}
		all += tmp;
		if (tmp>0.5*h)black_number++;
	}
	if (black_number < 0.3*w)return 0;

	white_number = 0;
	for (int i = h / 3; i < h - h / 3; i++){
		tmp = 0;
		for (int j = 0; j < w; j++)
		{
			if (Img.ptr<uchar>(i)[j] == 0)tmp++;
		}
		if (tmp<0.4*w)white_number++;
	}
	if (white_number < 0.05*h)return 0;

	ex_number = 0;
	tmp = 0;
	for (int i = 0; i < h; i++){
		int pre = tmp;
		tmp = 0;
		for (int j = 0; j < w; j++)
		{
			if (Img.ptr<uchar>(i)[j] == 0)tmp++;
		}
		if (pre - tmp>4 || pre - tmp < -4)ex_number++;
	}
	if (ex_number > 6)return 0;

	return 1;
}

int recsinglenumber(Rect item, Mat numImg, int index)
{
	int w = item.width;
	int h = item.height;
	double wbh = (double)w / (double)h;
	int N = 0;

	if (Is_One(numImg))return 1;
	//cout << w << " " << h << endl;
	//cout << numImg.cols << " " << numImg.rows << endl;
	if (wbh>WHB_THRESHOLD_DOWN&&wbh < WHB_THRESHOLD_UP){
		int count = 0;
		int number = 0;
		for (int i = max(0, (int)((double)h*NUMBER_INSIDE_H_FRAC));
			i < min((int)((double)h*((1 - 3.0*NUMBER_INSIDE_H_FRAC) / 2.0) + NUMBER_INSIDE_H_FRAC), h);
			i++)
		{

			for (int j = max(0, (int)((double)w*NUMBER_INSIDE_W_FRAC));
				j < min((int)((double)w*(1.0 - NUMBER_INSIDE_W_FRAC)), w);
				j++)

			{
				count++;
				if (numImg.ptr<uchar>(i)[j] == 0)number++;
			}
		}
		if ((double)number / (double)count > NUMBER_INSIDE_THRESHOLD)return -1;

		count = 0;
		number = 0;
		for (int i = max(0, (int)((double)h*((1 - 3.0*NUMBER_INSIDE_H_FRAC) / 2.0) + 2.0*NUMBER_INSIDE_H_FRAC));
			i < min((int)((double)h*(1 - NUMBER_INSIDE_H_FRAC)), h);
			i++)
		{

			for (int j = max(0, (int)((double)w*NUMBER_INSIDE_W_FRAC));
				j < min((int)((double)w*(1.0 - NUMBER_INSIDE_W_FRAC)), w);
				j++)

			{
				count++;
				if (numImg.ptr<uchar>(i)[j] == 0)number++;
			}
		}
		if ((double)number / (double)count > NUMBER_INSIDE_THRESHOLD)return -1;


		int all_count = 0;
		count = 0;
		number = 0;
		for (int i = 0; i < min((int)((double)h / 3.0), h); i++)
		{
			count = 0;
			number = 0;
			for (int j = 0; j < w; j++)
			{
				count++;
				if (numImg.ptr<uchar>(i)[j] == 0)number++;
			}
			if ((double)number / (double)count > NUMREC_THRESHOLD)all_count++;
		}
		if (all_count > COUNT_THRESHOLD)N |= HORIZ_UP;

		all_count = 0;
		count = 0;
		number = 0;
		for (int i = max(0, (int)((double)h / 3.0)); i < min((int)((double)h / 3.0 * 2), h); i++)
		{
			count = 0;
			number = 0;
			for (int j = 0; j < w; j++)
			{
				count++;
				if (numImg.ptr<uchar>(i)[j] == 0)number++;
			}
			if ((double)number / (double)count > NUMREC_THRESHOLD)all_count++;
		}
		if (all_count > COUNT_THRESHOLD)N |= HORIZ_MID;

		all_count = 0;
		count = 0;
		number = 0;
		for (int i = max(0, (int)((double)h / 3.0*2.0)); i < h; i++)
		{
			count = 0;
			number = 0;
			for (int j = 0; j < w; j++)
			{
				count++;
				if (numImg.ptr<uchar>(i)[j] == 0)number++;
			}
			if ((double)number / (double)count > NUMREC_THRESHOLD)all_count++;
		}
		if (all_count > COUNT_THRESHOLD)N |= HORIZ_DOWN;

		all_count = 0;
		count = 0;
		number = 0;
		for (int j = 0; j < min((int)((double)w / 2.0), w); j++)
		{
			count = 0;
			number = 0;
			for (int i = 0; i < min((int)((double)h / 2.0), h); i++)
			{
				count++;
				if (numImg.ptr<uchar>(i)[j] == 0)number++;
			}
			if ((double)number / (double)count > NUMREC_THRESHOLD)all_count++;
		}
		if (all_count > COUNT_THRESHOLD)N |= VERT_LEFT_UP;


		all_count = 0;
		count = 0;
		number = 0;
		for (int j = max(0, (int)((double)w / 2.0)); j < w; j++)
		{
			count = 0;
			number = 0;
			for (int i = 0; i < min((int)((double)h / 2.0), h); i++)
			{
				count++;
				if (numImg.ptr<uchar>(i)[j] == 0)number++;
			}
			if ((double)number / (double)count > NUMREC_THRESHOLD)all_count++;
		}
		if (all_count > COUNT_THRESHOLD)N |= VERT_RIGHT_UP;

		all_count = 0;
		count = 0;
		number = 0;
		for (int j = 0; j < min((int)((double)w / 2.0), w); j++)
		{
			count = 0;
			number = 0;
			for (int i = max(0, (int)((double)h / 2.0)); i < h; i++)
			{
				count++;
				if (numImg.ptr<uchar>(i)[j] == 0)number++;
			}
			if ((double)number / (double)count > NUMREC_THRESHOLD)all_count++;
		}
		if (all_count > COUNT_THRESHOLD)N |= VERT_LEFT_DOWN;



		all_count = 0;
		count = 0;
		number = 0;
		for (int j = max(0, (int)((double)w / 2.0)); j < w; j++)
		{
			count = 0;
			number = 0;
			for (int i = max(0, (int)((double)h / 2.0)); i < h; i++)
			{
				count++;
				if (numImg.ptr<uchar>(i)[j] == 0)number++;
			}
			if ((double)number / (double)count > NUMREC_THRESHOLD)all_count++;
		}
		if (all_count > COUNT_THRESHOLD)N |= VERT_RIGHT_DOWN;

		switch (N) {
		case D_ZERO: return 0;
		case D_ONE: return 1;
		case D_TWO: return 2;
		case D_THREE: return 3;
		case D_FOUR: return 4;
		case D_FIVE: return 5;
		case D_SIX: return 6;
		case D_SEVEN:
		case D_ALTSEVEN: return 7;
		case D_EIGHT: return 8;
		case D_NINE:
		case D_ALTNINE: return 9;
		default:
			 return -1;
		}
	}
	return -1;
}

bool cmp(const numberRec R1, const numberRec R2)
{
	return R1.pos_x > R2.pos_x;
}

bool recnumber(vector<Rect> item, vector<Mat> Imgs, int *x, int *y, int *z, int cols, int rows
#if defined(__linux__) && defined(LINUX_DEBUG)
	,const string image_tmp_path
#endif
	)
{
#if defined(__linux__) && defined(LINUX_DEBUG)
	string filepath = image_tmp_path + "/log2.txt";
	ofstream ofile(filepath.c_str());
#endif

	vector<numberRec> numbers;
	int i, j, k;
	int number;
	for (i = 0; i < Imgs.size(); i++)
	{
		if (item[i].width != 0){
			number = recsinglenumber(item[i], Imgs[i], i);

#if (defined(_WIN32)||defined(_WIN64)) && defined(OPENCV_SHOW)
			imshow("number_alone", Imgs[i]);
			cout << i << ":" << number << " position：" << item[i].x << "," << item[i].y << " size：" << item[i].width << "*" << item[i].height << endl;
			waitKey();
#endif
			if (number >= 0){
#if defined(__linux__) && defined(LINUX_DEBUG)
				ofile << i << ":" << number << " position：" << item[i].x << "," << item[i].y << " size：" << item[i].width << "*" << item[i].height << endl;
#endif

//#if (defined(_WIN32)||defined(_WIN64)) && defined(OPENCV_SHOW)
//				imshow("number_alone", Imgs[i]);
//				cout << i << ":" << number << " position：" << item[i].x << "," << item[i].y << " size：" << item[i].width << "*" << item[i].height << endl;
//				waitKey();
//#endif
				numberRec tmp;
				tmp.number = number;
				tmp.pos_x = item[i].x;
				tmp.pos_y = item[i].y;
				tmp.size_h = item[i].height;
				tmp.size_w = item[i].width;
				numbers.push_back(tmp);
			}
		}
	}
	sort(numbers.begin(), numbers.end(), cmp);
#if defined(__linux__) && defined(LINUX_DEBUG)
	for (i = 0; i < numbers.size(); i++)
	{
		ofile << i << ":" << numbers[i].number << " position：" << numbers[i].pos_x << "," << numbers[i].pos_y << " size：" << numbers[i].size_w << "*" << numbers[i].size_h << endl;
	}
#endif

#if (defined(_WIN32)||defined(_WIN64)) && defined(OPENCV_SHOW)
	for (i = 0; i < numbers.size(); i++)
	{
		cout << i << ":" << numbers[i].number << " position：" << numbers[i].pos_x << "," << numbers[i].pos_y << " size：" << numbers[i].size_w << "*" << numbers[i].size_h << endl;
	}
	waitKey();
#endif

	int n_rec = (int)numbers.size();
	int horizs[MAX_ITEM][MAX_ITEM];
	int horizs_x_pos[MAX_ITEM][MAX_ITEM];
	int horizs_number[MAX_ITEM];
	int horizs_y_pos[MAX_ITEM];
	int horizs_height[MAX_ITEM];
	int horizs_kind = 0;

	for (i = 0; i < MAX_ITEM; i++)
	{
		horizs_number[i] = 0;
		for (j = 0; j < MAX_ITEM; j++)
		{
			horizs_x_pos[i][j] = 99999;
		}
	}
	for (i = 0; i < n_rec; i++){
		int x_tmp = numbers[i].pos_x;
		int y_tmp = numbers[i].pos_y;
		int w_tmp = numbers[i].size_w;
		int h_tmp = numbers[i].size_h;
		int n_tmp = numbers[i].number;
		int flag = true;
		for (j = 0; j < horizs_kind; j++)
		{
			double sub_val = (double)(horizs_y_pos[j] - y_tmp) / (double)horizs_height[j];
			if (sub_val>-0.3 && sub_val < 0.3)
			{
				double sub_height = (double)(horizs_height[j] - h_tmp) / (double)horizs_height[j];
				if (sub_height>-0.25 && sub_height < 0.25){
					if (n_tmp != 1 && horizs_x_pos[j][0] - x_tmp > 2.5 * w_tmp)continue;
					if (n_tmp == 1 && horizs_x_pos[j][0] - x_tmp > 4.5 * w_tmp)continue;
					for (k = horizs_number[j] - 1; k >= 0; k--)
					{
						horizs[j][k + 1] = horizs[j][k];
						horizs_x_pos[j][k + 1] = horizs_x_pos[j][k];
					}
					horizs[j][0] = n_tmp;
					horizs_x_pos[j][0] = x_tmp;
					horizs_number[j]++;
					flag = false;
					break;
				}
			}
		}
		if (flag){
			horizs_y_pos[horizs_kind] = y_tmp;
			horizs_height[horizs_kind] = h_tmp;
			horizs[horizs_kind][0] = n_tmp;
			horizs_x_pos[horizs_kind][0] = x_tmp;
			horizs_number[horizs_kind]++;
			horizs_kind++;
		}
	}


	for (i = 0; i < horizs_kind; i++)
	{
		cout << "y_pos:" << horizs_y_pos[i] << " height:" << horizs_height[i] << " number:" << horizs_number[i] << endl;
		for (j = 0; j < horizs_number[i]; j++)
		{
			cout << horizs[i][j] << "(" << horizs_x_pos[i][j] << ")" << endl;
		}
		cout << endl << endl;
	}

#if defined(__linux__) && defined(LINUX_DEBUG)
	ofile.close();
#endif
	if (horizs_kind >= 3)
	{
		int max1_index = -1, max2_index = -1;
		int max1 = -1, max2 = -1;
		int max1_value = 0, max2_value = 0;
		for (i = 0; i < horizs_kind; i++)
		{
			int value_tmp = 0;
			if (horizs_number[i]>3)continue;
			for (j = 0; j<horizs_number[i]; j++)
			{
				value_tmp = value_tmp * 10 + horizs[i][j];
			}
			if (value_tmp>20 && value_tmp<300){
				if (horizs_height[i]>max1)
				{
					max2 = max1;
					max2_index = max1_index;
					max2_value = max1_value;
					max1 = horizs_height[i];
					max1_index = i;
					max1_value = value_tmp;
					continue;
				}
				if (horizs_height[i] > max2)
				{
					max2 = horizs_height[i];
					max2_index = i;
					max2_value = value_tmp;
				}
			}
		}

		if (max1_value > 300 || max1_value < 20 || max2_value>300 || max2_value < 20)return false;

		if (max1_value > max2_value)
		{
			*x = max1_value;
			*y = max2_value;
		}
		else
		{
			*x = max2_value;
			*y = max1_value;
		}


		for (i = 0; i < horizs_kind; i++)
		{
			if (horizs_number[i]>3)continue;
			if (i != max1_index && i != max2_index)
			{
				int value = 0;
				for (j = 0; j < horizs_number[i]; j++)
				{
					value = value * 10 + horizs[i][j];
				}
				if (value < 200 && value>20)
				{
					*z = value;
					break;
				}
			}
		}
		return true;
	}


	return false;
}

DLL_API bool find_four_points(void* img, void* mem_storage, int *ax, int *ay, int *bx, int *by, int *cx, int *cy, int *dx, int *dy)
{
	CvMemStorage* storage = (CvMemStorage*)mem_storage;
	CvSeq* rec = findSquares4((IplImage*)img, storage, ((IplImage*)img)->imageSize);
	if (rec == NULL){
		*ax = -1;
		*ay = -1;
		*bx = -1;
		*by = -1;
		*cx = -1;
		*cy = -1;
		*dx = -1;
		*dy = -1;
		return false;
	}
	CvSeqReader reader;
	cvStartReadSeq(rec, &reader, 0);
	CvPoint pt[6];
	CV_READ_SEQ_ELEM(pt[0], reader);
	CV_READ_SEQ_ELEM(pt[1], reader);
	CV_READ_SEQ_ELEM(pt[2], reader);
	CV_READ_SEQ_ELEM(pt[3], reader);
	int tmp = LONG_MAX, ancor;
	for (int i = 0; i < 4; i++){
		if (pt[i].x + pt[i].y < tmp) { tmp = pt[i].x + pt[i].y; ancor = i; }
	}

	pt[4] = pt[0]; pt[5] = pt[0];

	for (int i = 0; i < 3; i++){
		for (int j = i + 1; j < 4; j++){
			if (pt[i].x>pt[j].x){
				int tmp_x, tmp_y;
				tmp_x = pt[i].x;
				tmp_y = pt[i].y;
				pt[i].x = pt[j].x;
				pt[i].y = pt[j].y;
				pt[j].x = tmp_x;
				pt[j].y = tmp_y;
			}
		}
	}

	if (pt[0].y > pt[1].y){
		int tmp_x, tmp_y;
		tmp_x = pt[0].x;
		tmp_y = pt[0].y;
		pt[0].x = pt[1].x;
		pt[0].y = pt[1].y;
		pt[1].x = tmp_x;
		pt[1].y = tmp_y;
	}

	if (pt[2].y > pt[3].y){
		int tmp_x, tmp_y;
		tmp_x = pt[2].x;
		tmp_y = pt[2].y;
		pt[2].x = pt[3].x;
		pt[2].y = pt[3].y;
		pt[3].x = tmp_x;
		pt[3].y = tmp_y;
	}

	*ax = pt[0].x; *ay = pt[0].y;
	*bx = pt[1].x; *by = pt[1].y;
	*cx = pt[2].x; *cy = pt[2].y;
	*dx = pt[3].x; *dy = pt[3].y;

	cout << "a:" << *ax << " " << *ay << endl;

	cout << "b:" << *bx << " " << *by << endl;

	cout << "c:" << *cx << " " << *cy << endl;

	cout << "d:" << *dx << " " << *dy << endl;

	if (*ax < 0 || *ay < 0 || *ax>1288 || *ay>1288)return false;
	if (*bx < 0 || *by < 0 || *bx>1288 || *by>1288)return false;
	if (*cx < 0 || *cy < 0 || *cx>1288 || *cy>1288)return false;
	if (*dx < 0 || *dy < 0 || *dx>1288 || *dy>1288)return false;
	return true;
}

DLL_API bool find_rect_Rec(void* img, int ax, int ay, int bx, int by, int cx, int cy, int dx, int dy, int *high, int *low, int *heart)
{
#if defined(__linux__) && defined(LINUX_DEBUG)
	time_t nowtime;
	struct tm *timeinfo;
	time(&nowtime);
	timeinfo = localtime(&nowtime);
	ostringstream s1;
	s1 << "/tmp/numrec/time-" << timeinfo->tm_hour << "-" << timeinfo->tm_min << "-" << timeinfo->tm_sec;
	const string image_tmp_path = s1.str();

	if (access(image_tmp_path.c_str(), 0777) == -1)
	{
		mkdir(image_tmp_path.c_str(), 0777);
	}
#endif
	if (ax < 0 || ay < 0 || ax>1288 || ay>1288)return false;
	if (bx < 0 || by < 0 || bx>1288 || by>1288)return false;
	if (cx < 0 || cy < 0 || cx>1288 || cy>1288)return false;
	if (dx < 0 || dy < 0 || dx>1288 || dy>1288)return false;

	Point2d a(ax, ay);
	Point2d b(bx, by);
	Point2d c(cx, cy);
	Point2d d(dx, dy);

	//cout<<hex<<"yes"<<(long long)img<<endl;

	//仿射变换到固定大小
	//Mat imagepic(imgnew, 0);
	//imshow("imgnew", imagepic);
	Mat imagepic((IplImage*)img, 0);
	int w_a4 = 700, h_a4 = 800;
	Mat dst = Mat::zeros(h_a4, w_a4, CV_8UC3);

	vector<Point2f> dst_pts, img_pts;

	dst_pts.push_back(Point(0, 0));
	dst_pts.push_back(Point(w_a4 - 1, 0));
	dst_pts.push_back(Point(0, h_a4 - 1));
	dst_pts.push_back(Point(w_a4 - 1, h_a4 - 1));


	img_pts.push_back(a);
	img_pts.push_back(c);
	img_pts.push_back(b);
	img_pts.push_back(d);

	Mat transmtx = getPerspectiveTransform(img_pts, dst_pts);

	Mat mid0 = imagepic;
	//imshow("mid0原图", mid0);
	warpPerspective(imagepic, mid0, transmtx, dst.size());
	//imshow("mid0矩阵变换后图", mid0);

	//將取出的屏幕二值化
	Mat kern = (Mat_<char>(3, 3) << 0, -2, 0,
		-2, 10, -2,
		0, -2, 0);


	Mat mid1, mid2, mid3, out, mid4;
	cvtColor(mid0, mid1, CV_RGB2GRAY);
	//imshow("mid1", mid1);
	filter2D(mid1, mid2, mid1.depth(), kern);
	//imshow("mid2", mid2);
	equalizeHist(mid2, mid3);
	//imshow("mid3", mid3);
	//threshold(mid3, out, 40, 255, 0);
	adaptiveThreshold(mid1, out, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 101, 5);
	//imshow("out", out);

	int count;
	vector<Rect> items(40);
	vector<Mat> itemImg;
	out.copyTo(mid4);
	count = findItems(out, items
#if defined(__linux__) && defined(LINUX_DEBUG)
		,image_tmp_path
#endif
		);

	//adaptiveThreshold(mid2, mid4, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 101, 35);

	//imshow("原图二值化", mid4);
	//imshow("使用的图", mid4);
	for (int i = 0; i < 40; i++){

		if (items[i].width != 0) {
			itemImg.push_back(mid4(items[i]));
#if defined(__linux__) && defined(LINUX_DEBUG)
			ostringstream picture_lonely_path;
			picture_lonely_path<<image_tmp_path<<"/picture"<<i << ".jpg";
			string name = picture_lonely_path.str();
			imwrite(name, itemImg[i]);
#endif
		}
	}

#if defined(__linux__) && defined(LINUX_DEBUG)
	string logfile = image_tmp_path+"/log.txt";
	ofstream ofile(logfile.c_str());
	for (int i = 0; i < 40; i++){
		if (items[i].width != 0){
			ofile << i << " " << items[i].width << " " << items[i].height << " " << (double)items[i].width / (double)items[i].height
				<< " " << itemImg[i].cols << " " << itemImg[i].rows << endl;
			for (int k = 0; k < itemImg[i].rows; k++){
				for (int j = 0; j < itemImg[i].cols; j++)
				{
					if ((int)itemImg[i].ptr<uchar>(k)[j] == 0)ofile << "1 ";
					else ofile << "  ";
				}
				ofile << endl;
			}
		}
	}
	ofile.close();
#endif

#if defined(_WIN32) && defined(_WIN64)
	//waitKey(0);
#endif

	int x = -1, y = -1, z = -1;
	if (recnumber(items, itemImg, &x, &y, &z, out.cols, out.rows
#if defined(__linux__) && defined(LINUX_DEBUG)
		,image_tmp_path
#endif
		)){
		*high = x; *low = y; *heart = z;
		return true;
	}
	else
		return false;
}

DLL_API bool openfile(const char* filename, void** openpicture, void** mem_storage){

#if defined(__linux__) && defined(LINUX_DEBUG)
	if (access("/tmp/numrec", 0777) == -1)
	{
		mkdir("/tmp/numrec", 0777);
	}
#endif

#if defined(__linux__)
	struct sysinfo s_info;
	sysinfo(&s_info);
	cout << "remain memory:" << s_info.freeram<<endl;
	if (s_info.freeram < 80000000)return false;

#endif
	IplImage *img0 = NULL;
	img0 = cvLoadImage(filename, 1);
	if (img0 == NULL){
		cout<<"error:img read fault"<<endl;
		return false;
	}
	*mem_storage = (void*)cvCreateMemStorage(0);
	if (*mem_storage == NULL){
		cout<<"error: mem create fault"<<endl;
		return false;
	}
	IplImage *imgnew = cvCreateImage(cvSize(1288, 1288), IPL_DEPTH_8U, 3);
	cvResize(img0, imgnew, CV_INTER_LINEAR);
	cvReleaseImage(&img0);
	*openpicture = (void*)imgnew;
	//cout<<"imgnew"<<(long long)imgnew<<endl;
	return true;
}


DLL_API void closefile(void** picture, void** mem_storage){
	if (picture != NULL && *picture != NULL)cvReleaseImage((IplImage**)picture);
	if (mem_storage != NULL && *mem_storage != NULL)cvReleaseMemStorage((CvMemStorage**)mem_storage);
}


