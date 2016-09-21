#pragma once

//该宏完成在dll项目内部使用__declspec(dllexport)导出  
//在dll项目外部使用时，用__declspec(dllimport)导入  
//宏DLL_IMPLEMENT在SimpleDLL.cpp中定义
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