#pragma once

//�ú������dll��Ŀ�ڲ�ʹ��__declspec(dllexport)����  
//��dll��Ŀ�ⲿʹ��ʱ����__declspec(dllimport)����  
//��DLL_IMPLEMENT��SimpleDLL.cpp�ж���
#if defined(_WIN32) || defined(_WIN64) 
#ifdef DLL_IMPLEMENT  
#define DLL_API __declspec(dllexport)  
#else  
#define DLL_API __declspec(dllimport)  
#endif  
#else
#ifdef __linux__
#define DLL_API
#endif
#endif

DLL_API bool find_four_points(void* picutre, void* mem_storage, int *ax, int *ay, int *bx, int *by, int *cx, int *cy, int *dx, int *dy);
DLL_API bool find_rect_Rec(void* picture, int ax, int ay, int bx, int by, int cx, int cy, int dx, int dy, int *high, int *low, int *heart);
DLL_API bool openfile(const char* filename, void** picture, void** mem_storage);
DLL_API void closefile(void** picture, void** mem_storage);