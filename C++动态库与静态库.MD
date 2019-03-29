# C++动态库与静态库

***

## 一. 基础知识

动态链接库（dll）比较好的介绍在孙鑫《VC++深入详解》第十九章动态链接库中有详细介绍。主要涉及以下问题：

 1. [静态链接库与动态链接库的区别](https://blog.csdn.net/zang141588761/article/details/50501914)

    (1) 静态链接库编译的时候加载；  
    (2) 动态链接库运行的时候加载。  

 2. [dll的显式调用和隐式调用](https://blog.csdn.net/sinat_22991367/article/details/73694764)

    (1) dll隐式加载：需要.h，.lib，.dll文件  
    (2) dll显式加载：只需要.dll文件

 3. [dll名字改编问题](https://blog.csdn.net/aidem_brown/article/details/38712705)

 4. [动态链接库导出类与导出函数](https://zhidao.baidu.com/question/941508810337656732.html)
***

## 二、SCR5逆运动算法C++ DLL封装

1. 打开VS打开VS2015,创建一个Win32项目，命名为Ikine_analyze_dll

2. 添加Ikine_analyze_dll.h和Ikine_analyze_dll.cpp文件

3. 在Ikine_analyze_dll.h中编写SCR5逆运动算法函数接口，并添加dll名字改编

4. 在Ikine_analyze_dll.cpp中编写SCR5逆运动算法函数实现

5. 配置Debug或者Release以及平台X86和X64

6. 点击生成->生成解决方案，于是创建完成dll
***
## win32控制台中的使用方法

采用dll隐式调用方法调用dll。

1. 创建win32控制台程序Test
2. 将Ikine_analyze_dll.h和Ikine_analyze_dll.lib文件拷贝到工程目录下
3. 添加.h包含目录和lib库目录
4. 添加附加依赖项Ikine_analyze_dll.lib
5. 编写main代码，其中若添加#pragma comment(lib,"Ikine_analyze_dll.lib")，则不需要进行3和4的包含及库目录配置
6. 运行程序出现错误
7. 将Ikine_analyze_dll.dll文件拷贝到debug或者release文件夹下，再次运行