An Open-Source Python GNSS SDR Toolbox with SIMD/GPU Acceleration： pySoftGNSS 
===============================================================================



Overview
-------------------------------------------------------------------------------
pySoftGNSS是一个基于Python的开源工具箱，用于对录制的GNSS中频（IF）信号进行后处理。
它采用与mSoftGNSS一致的接收机架构，以Python模块和类组织信号处理功能。
工具箱支持GPS、Galileo、GLONASS和北斗信号，提供捕获、数据/导频跟踪、导航电文解码、
伪距计算和定位功能，具体处理能力因信号而异。CPU和GPU捕获分别使用NumPy/SciPy和CuPy。
通道串行和通道并行两种跟踪模式均支持NumPy相关器、SIMD加速相关器和GPU加速相关器。
SIMD包含x86 AVX2和ARM64 NEON两种实现，由程序根据运行平台自动选择。
通道并行跟踪中，各通道共用同一中频数据缓冲区，并批量执行相关任务。C++和CUDA C++相关器编译为
动态链接库，通过ctypes.CDLL调用；调度、跟踪环路控制、导航解码和定位仍在Python中完成。
这种设计兼顾上层代码的可读性与参数配置的灵活性，加速了计算密集的采样点级运算。
工具箱还支持现代化GNSS信号和LDPC解码，并提供结果可视化功能，可用于GNSS算法研究、
接收机原型开发和可复现的实验评估。



Authors
-------------------------------------------------------------------------------
* Yafeng Li
    * E-Mail: <lyf8118@126.com>
    * Wechat: lyf8118521
    * GNSS软件接收机技术讨论QQ群：147304049


* Dennis Akos  
    * E-Mail: <dma@colorado.edu>
    * HP: <http://www.colorado.edu/aerospace/dennis-akos>




Features
-------------------------------------------------------------------------------
* 基于 Python 实现的 GNSS 信号处理功能
    * 本地码生成
    * CPU 和 GPU 捕获
    * 通道串行和通道并行跟踪
    * Python/NumPy、SIMD 动态库和 CUDA 动态库相关器
    * 现代化信号的数据/导频跟踪
    * 载噪比（C/N0）估计和跟踪结果绘图
    * 导航电文解码（包括 LDPC 解码器）
    * 伪距生成
    * 接收机位置与钟差解算及结果绘图
* SIMD/GPU 加速和相关器选项
    * settings.gpuACQflag = 0/1 分别选择 CPU/GPU 捕获。
    * settings.trkMode = 0/1 分别选择通道串行/通道并行跟踪。
    * settings.correlatorType = 0 表示使用 Python/NumPy 脚本相关器。
    * settings.correlatorType = 1 表示使用 CPU SIMD 动态库相关器。
    * settings.correlatorType = 2 表示使用 CUDA GPU 动态库相关器。
    * Python/NumPy 脚本相关器和各信号的本地码生成函数，
      位于各接收机的 correlator.py 中。
    * 共用相关器库包含 BPSK、QPSK 和 QMBOC 的
      SIMD 和 CUDA 实现，通过 ctypes.CDLL 调用。
* 支持的信号
    * GPS L1 C/A （数据）
    * GPS L1C （数据与导频）
    * GPS L2C （数据与导频）
    * GPS L5 （数据与导频）
    * Galileo E1 （数据与导频）
    * Galileo E5a （数据与导频）
    * Galileo E5b （数据与导频）
    * Galileo E6 （数据与导频；HAS 页解码）
    * GLONASS L1OF （数据）
    * GLONASS L2OF （数据）
    * GLONASS L1OC （数据与导频）
    * GLONASS L2OCp （导频）
    * GLONASS L3OC （数据与导频）
    * 北斗 B1I （数据）
    * 北斗 B3I （数据）
    * BDS-3 B1C （数据与导频）
    * BDS-3 B2a （数据与导频）
    * 北斗 B2b （数据）
* 支持以二进制文件存储的中频采样数据
    * settings.dataType 可设为 int8 或 int16，两种类型均支持实数和复数采样。
    * settings.fileType = 1 表示实数采样，2 表示 I/Q 交织复数采样
      （I0,Q0,I1,Q1,...）；dataType 指定每个 I/Q 分量的数据类型。
    * 下文提供 LimeSDR 复数采样和 NUT4NT 实数采样数据的配置示例；
      使用其他数据时，应根据录制参数设置数据格式、中频和采样率。
    * 加速相关器按目标 CPU 和操作系统编译，通过 ctypes 加载，
      不需要 MATLAB。



Directory and Files
-------------------------------------------------------------------------------
根目录

    ./Doc                     文档、ICD、论文和各接收机说明材料。
    ./IF_Data_Set             可放置待处理 IF 数据及对应元数据文件。
    ./native_Correlators      所有接收机共用的 SIMD/CUDA 相关器源码和
                              在本机编译的动态库文件。

接收机目录

    每个接收机目录都是相对独立的软件接收机；除 ./native_Correlators、
    ./IF_Data_Set 和 ./Doc 外，各接收机代码互不依赖。当前接收机目录如下：

    ./pyGPS_L1CA              GPS L1 C/A 软件接收机
    ./pyGPS_L1C               GPS L1C 软件接收机
    ./pyGPS_L2C               GPS L2C 软件接收机
    ./pyGPS_L5                GPS L5 软件接收机
    ./pyGalileo_E1            Galileo E1 软件接收机
    ./pyGalileo_E5a           Galileo E5a 软件接收机
    ./pyGalileo_E5b           Galileo E5b 软件接收机
    ./pyGalileo_E6            Galileo E6B/E6C 软件接收机（HAS 接收）
    ./pyGlonass_L1_L2         GLONASS L1OF/L2OF 软件接收机
    ./pyGlonass_L1OC          GLONASS L1OC 软件接收机
    ./pyGlonass_L2OC          GLONASS L2OCp 软件接收机
    ./pyGlonass_L3OC          GLONASS L3OC 软件接收机
    ./pyBDS_B1I               北斗 B1I软件接收机
    ./pyBDS_B3I               北斗 B3I 软件接收机
    ./pyBDS-3_B1C             北斗三号 B1C 软件接收机
    ./pyBDS-3_B2a             北斗三号 B2a 软件接收机
    ./pyBDS_B2b               北斗 B2b 软件接收机

每个接收机目录的通用结构

    ./main.py                 接收机启动脚本：切换到接收机目录、读取配置、绘制中频
                              信号的时域波形和频谱，并询问是否启动处理。
    ./initSettings.py         Settings 类：接收机参数配置、probeData() 和顶层
                              postProcessing() 处理调度。
    ./acquisition.py          AcqEngine 类：CPU/CuPy GPU 捕获、结果整理和捕获绘图。
    ./tracking.py             Channel 和 TrackingEngine 类：通道串行/并行跟踪和绘图。
    ./postNavigation.py       Ephemeris、NavSolutions 和 NavigationEngine 类：
                              导航电文解码和定位解算。
    ./correlator.py           本地码生成/采样、Python/NumPy 相关器和 ctypes 动态库接口类。
    ./channelDecoder.py       LDPC 解码器及相关 BCH 解码器类；仅存在于
                              pyBDS-3_B1C、pyBDS-3_B2a、pyBDS_B2b 和 pyGPS_L1C。
    ./commUtils.py            本接收机使用的通用数学与定位函数：伪距计算、
                              坐标转换、最小二乘定位、环路参数计算和进度显示。
    ./result_Cache            保存捕获、跟踪和导航/星历结果的 .pkl 文件，请保留该目录。
                              Galileo E1 还需保留其中的 E1b.dat/E1c.dat；
                              E6B 还需保留其中的 E6B_codes.dat/E6C_codes.dat。
    ./assets                  Galileo E6B HAS 的 Reed-Solomon 生成矩阵，
                              位于 pyGalileo_E6 中。

./native_Correlators 中的共用相关器

    corrX86SIMDSerial*.cpp      x86 AVX2 通道串行跟踪相关器源码。
    corrX86SIMDParallel*.cpp    x86 AVX2 通道并行跟踪相关器源码。
    corrNeonSIMD*.cpp           ARM64 NEON 通道串行/并行相关器源码。
    corrGPUSerial*.cu           CUDA 通道串行跟踪相关器源码。
    corrGPUParallel*.cu         CUDA 通道并行跟踪相关器源码。
    corrGPUParallelFused*.cu    CUDA 融合内核通道并行跟踪相关器源码。
    *.dll / *.so                在本机编译的动态库，由 settings.correlatorType 选择调用。

相关器选择

    settings.correlatorType = 0
                              使用各接收机 correlator.py 中的 Python/NumPy 脚本相关器，
                              便于阅读和调试。
    settings.correlatorType = 1
                              使用 ./native_Correlators 中的 CPU SIMD 动态库相关器，
                              程序根据运行平台架构自动选择 x86 或 ARM64 NEON，
                              不依赖 CUDA 即可加速跟踪。
    settings.correlatorType = 2
                              使用 ./native_Correlators 中的 CUDA GPU 动态库相关器，
                              用于高吞吐的通道串行或通道并行跟踪。

Software Dependencies
-------------------------------------------------------------------------------
* 需要 Python 3.10 或更高版本，以及 NumPy、SciPy 和 Matplotlib。
  原生相关器需要架构匹配的 64 位 Python 与动态库。
* SciPy 提供信号处理和数值计算工具；Matplotlib 用于 IF 频谱和接收机结果绘图。
* 导航解码在 postNavigation.py 中实现；四个使用 LDPC 的接收机还使用
  channelDecoder.py，不需要 MATLAB 工具箱。
* GPU 捕获需要 CuPy、兼容的 CUDA 环境及 NVIDIA GPU/驱动。
  CUDA 动态库相关器需要兼容的 NVIDIA GPU 和驱动；
  动态库调用本身不依赖 CuPy。
* x86 SIMD 需要 AVX2 和 FMA；ARM64 使用 NEON 源码。
  SIMD 编译需要 C++ 编译器；CUDA 编译需要 nvcc 和受支持的主机编译器。
  编译生成的.dll和.so文件直接放在 native_Correlators 中。
* 跟踪进度窗口使用 Tkinter，Python 环境中需提供该组件。
* 如果只需要 CPU 运行，可在各接收机 initSettings.py 中关闭 GPU 捕获，并选择
  Python/NumPy 相关器或 SIMD 相关器。
  
  

Compile correlators
-------------------------------------------------------------------------------
本节仅说明 pySoftGNSS 的原生相关器编译方法，请按操作系统和 CPU 架构选择。

命令在系统终端执行，初始目录为 pySoftGNSS 根目录，只执行对应平台的章节。
所需编译器应在 PATH 中可用；重新编译前先退出正在使用这些动态库的 Python 进程。

| 平台 | SIMD 要求 | 编译产物 |
| --- | --- | --- |
| Windows x86-64 | 支持 AVX2 和 FMA 的 x86-64 CPU | .dll |
| Linux x86-64 | 支持 AVX2 和 FMA 的 x86-64 CPU | .so |
| Linux ARM64 | 支持 NEON 的 AArch64 CPU | .so |

进入native_Correlators 后，选择对应架构的SIMD源码和CUDA源码，Python 通过 ctypes 调用 C 接口

SIMD 编译只需要对应平台的 C++ 编译器。CUDA 编译需要 CUDA Toolkit 及 nvcc 支持的主机编译器；
运行 GPU 相关器时还需要兼容的 NVIDIA GPU 和驱动。

下面用 cudaArch / CUDA_ARCH 指定目标 GPU 的计算能力。Windows 和 Linux x86-64 以 RTX 4080 Laptop GPU 的 '89' 为例；

Linux ARM64 以 Jetson Orin NX 的 '87' 为例，换用其他 GPU 时必须按实际计算能力修改，并确认所用 nvcc 支持该目标；

### Windows x86-64

SIMD 以 64 位 MinGW-w64/G++ 为例，在 PowerShell 中执行。
"g++ -dumpmachine" 应显示 x86_64 Windows 编译目标，AVX2 和 FMA 两项都要启用。

```powershell
Set-Location .\native_Correlators
Get-ChildItem -File 'corrX86SIMD*.cpp' | ForEach-Object {
    & g++ -O3 -std=c++14 -mavx2 -mfma -shared -static-libgcc -static-libstdc++ $_.FullName -o "$($_.BaseName).dll"
    if ($LASTEXITCODE -ne 0) { throw "Build failed: $($_.Name)" }
}
```

CUDA 使用 nvcc + Microsoft Visual C++。先初始化兼容的 x64 MSVC 环境，例如在
Visual Studio 的 "x64 Native Tools Command Prompt" 中启动 PowerShell。
进入同一 native_Correlators 目录后执行，并按 GPU 修改 cudaArch；

```powershell
$cudaArch = '89'
Get-ChildItem -File 'corrGPU*.cu' | ForEach-Object {
    & nvcc -O3 -std=c++14 --shared --cudart static -Xcompiler /MD -Xcompiler /utf-8 "-gencode=arch=compute_$cudaArch,code=sm_$cudaArch" $_.FullName -o "$($_.BaseName).dll"
    if ($LASTEXITCODE -ne 0) { throw "Build failed: $($_.Name)" }
}
```

完成 SIMD 和 CUDA 编译后，本目录中生成.dll 文件。CUDA 运行库采用静态链接，但仍需要 NVIDIA 驱动和 Microsoft C++ 运行库。

### Linux x86-64

使用原生 x86-64 GCC/G++，在 Bash 中编译 SIMD：

```bash
cd native_Correlators
(
    set -e
    for src in corrX86SIMD*.cpp; do
        g++ -O3 -std=c++14 -mavx2 -mfma -fPIC -shared \
            "$src" -o "${src%.cpp}.so"
    done
)
```

CUDA 使用 nvcc 和它支持的 GCC/G++，两者都应在 PATH 中可用。
在同一 native_Correlators 目录执行，按实际 GPU 修改 CUDA_ARCH：

```bash
(
    set -e
    CUDA_ARCH=89
    for src in corrGPU*.cu; do
        nvcc -O3 -std=c++14 --shared --cudart static -ccbin g++ -Xcompiler -fPIC \
            "-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}" \
            "$src" -o "${src%.cu}.so"
    done
)
```

完成编译后，本目录中生成 x86 SIMD .so 和 CUDA .so。

### Linux ARM64

使用 ARM64 原生 Python 和 AArch64 GCC/G++，在 ARM64 机器的 Bash 中执行，SIMD 使用独立的 NEON 源码。

```bash
cd native_Correlators
(
    set -e
    for src in corrNeonSIMD*.cpp; do
        g++ -O3 -std=c++17 -march=armv8-a+simd -ffp-contract=off -fPIC -shared \
            "$src" -o "${src%.cpp}.so"
    done
)
```

编译 CUDA 相关器需要对应的 ARM64 CUDA 工具链；运行时还需要兼容的 NVIDIA GPU 和驱动。
在同一 native_Correlators 目录执行，按实际 GPU 修改 CUDA_ARCH。
保留 -ffp-contract=off 和 --fmad=false，以保持该 ARM 实现的数值编译设置。

```bash
(
    set -e
    CUDA_ARCH=87
    for src in corrGPU*.cu; do
        nvcc -O3 -std=c++17 -arch="sm_${CUDA_ARCH}" --fmad=false \
            --shared --cudart static -ccbin g++ -Xcompiler -fPIC "$src" -o "${src%.cu}.so"
    done
)
```

完成 SIMD 和 CUDA 编译后，本目录中生成 NEON .so 和 CUDA .so。



How to use
-------------------------------------------------------------------------------
* Step 1: 进入目标信号对应的接收机目录，例如 "pyBDS-3_B1C" 或 "pyGPS_L1CA"。
* Step 2: 将 IF 数据放入 "IF_Data_Set"，或在 settings.fileName 中设置数据文件
          的绝对路径。
* Step 3: 在 "initSettings.py" 中配置处理时长、IF 数据、捕获、跟踪、相关器和
          导航解算参数。skipNumberOfSamples 按采样点计数，一个实数值或一对 I/Q
          分量计为一个采样点。
* Step 4: 通过 settings.gpuACQflag 选择 CPU/GPU 捕获，通过 settings.trkMode
          选择通道串行/并行跟踪，通过 settings.correlatorType 选择脚本、SIMD
          或 GPU 相关器。
* Step 5: 在当前接收机目录下运行 "python main.py"，查看中频数据图后，
          输入 1 开始处理，或输入 0 退出。
* Step 6: 处理完成后，查看捕获、跟踪和导航定位结果及相关图形。



Implementation details
-------------------------------------------------------------------------------
本软件接收机套件遵循 SoftGNSS 的处理架构，以 Python 模块和类组织实现，
支持多种现代化 GNSS 信号、CPU/GPU 捕获、通道串行/并行跟踪，以及通过
ctypes.CDLL 调用的共用 SIMD/CUDA 相关器库。

通道并行跟踪以历元为外层循环，通过共享中频数据缓冲区和批量相关器调用，集中完成
各活动通道的相关运算，再分别更新各通道的跟踪环路。该架构在保留各通道独立状态
与环路反馈的同时，减少中频数据的重复读取和接口调用开销，并支持 SIMD/GPU
多通道批量加速。

BDS-3 B1C/B2a 接收机设计可参考：
Li, Y., Shivaramaiah, N.C. & Akos, D.M. Design and implementation of an open-source
BDS-3 B1C/B2a SDR receiver. GPS Solut (2019) 23: 60.
https://doi.org/10.1007/s10291-019-0853-z


​	   
Test signal (collected by LimeSDR) and parameter configurations
-------------------------------------------------------------------------------
    * L1CA_E1_B1C.bin
       -- for GPS L1 C/A, Galileo E1 and BDS B1C 
       -- dataType: int16
       -- fileType: 2（I/Q 交织复数采样，每个分量为 int16）
       -- IF: 0 Hz
       -- samplingFreq: 30e6 Hz
       -- link: https://pan.baidu.com/s/1NdZeVrEdEjDnJiV91Cg0qg?pwd=bvh5 password: bvh5
    * GPS L1C.iq
       -- for GPS L1C 
       -- dataType: int8
       -- fileType: 2（I/Q 交织复数采样，每个分量为 int8）
       -- IF: 0 Hz
       -- samplingFreq: 25e6 Hz
       -- link: https://pan.baidu.com/s/1hZ4S8R4VWB5OtPr9VyPCYQ?pwd=tv8b password: tv8b
    * GPS_L2.bin
       -- for GPS L2C
       -- dataType: int16
       -- fileType: 2（I/Q 交织复数采样，每个分量为 int16）
       -- IF: 0 Hz
       -- samplingFreq: 30e6 Hz
       -- link: https://pan.baidu.com/s/1KqdyWyuVPAgx0EoUdoLwag?pwd=5tya password: 5tya
    * L5_E5a_B2a.bin
       -- for GPS L5, Galileo E5a and BDS B2a
       -- dataType: int16
       -- fileType: 2（I/Q 交织复数采样，每个分量为 int16）
       -- IF: 0 Hz
       -- samplingFreq: 30e6 Hz
       -- link: https://pan.baidu.com/s/1rX2lguJgMmgjAdZIJK1FNg?pwd=bnzm password: bnzm
    
    * E5b_B2b.bin   
       -- for Galileo E5b, BDS B2b
       -- dataType: int16
       -- fileType: 2（I/Q 交织复数采样，每个分量为 int16）
       -- IF: 0 Hz
       -- samplingFreq: 30e6 Hz
       -- link: https://pan.baidu.com/s/128dwHIKUG9iM0MigS_7vJw?pwd=x4uq password: x4uq
    * GAL_E6.bin  
       -- for Galileo E6
       -- dataType: int16
       -- fileType: 2（I/Q 交织复数采样，每个分量为 int16）
       -- IF: 0 Hz
       -- samplingFreq: 30e6 Hz
       -- link: https://pan.baidu.com/s/1OrHtpNG-dQxSmmVc9rX6tA?pwd=qfx8 password: qfx8
    * GLO_L1OC_L1.bin 
       -- for GLONASS L1OF  
       -- dataType: int16
       -- fileType: 2（I/Q 交织复数采样，每个分量为 int16）
       -- IF: 1.005e6 Hz
       -- samplingFreq: 30e6 Hz
       -- GLONASS L1OC
       -- dataType: int16
       -- fileType: 2（I/Q 交织复数采样，每个分量为 int16）
       -- IF: 0 Hz
       -- samplingFreq: 30e6 Hz
       -- link: https://pan.baidu.com/s/1w-fYtmM3oW_pxR4ZH_HGyw?pwd=js93 password: js93
    * GLO_L2OC_L2.bin  
       -- for GLONASS L2OF 
       -- dataType: int16
       -- fileType: 2（I/Q 交织复数采样，每个分量为 int16）
       -- IF: -2.06e6 Hz
       -- samplingFreq: 30e6 Hz
    
       -- for GLONASS L2OC 
       -- dataType: int16
       -- fileType: 2（I/Q 交织复数采样，每个分量为 int16）
       -- IF: 0 Hz
       -- samplingFreq: 30e6 Hz
       -- link: https://pan.baidu.com/s/1jefXiCkeI9qHSqwZ_RXr2g?pwd=33w6 password: 33w6
    * GLO_L3OC.bin  
       -- for GLONASS L3OC	
       -- dataType: int16
       -- fileType: 2（I/Q 交织复数采样，每个分量为 int16）
       -- IF: 0 Hz
       -- samplingFreq: 30e6 Hz
       -- link: https://pan.baidu.com/s/1ySNHn3W66Q6n9fxERR7nOQ?pwd=289m password: 289m
    * BDS_B1I.bin
       -- for BDS B1I 
       -- dataType: int16
       -- fileType: 2（I/Q 交织复数采样，每个分量为 int16）
       -- IF: 0 Hz
       -- samplingFreq: 30e6 Hz
       -- link: https://pan.baidu.com/s/1qV-TCtROGHL8iJZlqw5mZA?pwd=39vd password: 39vd
    * BDS_B3I.bin       
       -- for BDS B3I    
       -- dataType: int16
       -- fileType: 2（I/Q 交织复数采样，每个分量为 int16）
       -- IF: 0 Hz
       -- samplingFreq: 30e6 Hz
       -- link: https://pan.baidu.com/s/1T_g1to1e8OW5OFtHlQsZeg?pwd=6w1u password: 6w1u 



Test signal (collected by NUT4NT) and parameter configurations
-------------------------------------------------------------------------------
    * L1CA_E1_B1C_real.bin
       -- for GPS L1 C/A, Galileo E1 and BDS B1C
       -- dataType: int8
       -- fileType: 1（实数采样）
       -- IF: -14.58e6 Hz
       -- samplingFreq: 53e6 Hz
       -- link: https://pan.baidu.com/s/1cFGCBwcWA0vHBeiL7WU_2Q?pwd=r8au password: r8au
    * B1I_real.bin
       -- for BDS B1I
       -- dataType: int8
       -- fileType: 1（实数采样）
       -- IF: -28.902e6 Hz
       -- samplingFreq: 99.375e6 Hz
       -- link: https://pan.baidu.com/s/1y4LfFZSO-Qm3Tvshvg_Gzg?pwd=86ii password: 86ii
    * GLO_L1OF_L1OC_real.bin
       -- for GLONASS L1OF
       -- dataType: int8
       -- fileType: 1（实数采样）
       -- IF: +12e6 Hz
       -- samplingFreq: 53e6 Hz
    
       -- for GLONASS L1OC
       -- dataType: int8
       -- fileType: 1（实数采样）
       -- IF: +10.995e6 Hz
       -- samplingFreq: 53e6 Hz
       -- link: https://pan.baidu.com/s/1-fxXe6VW36izl2ve3dMQsw?pwd=j7p9 password: j7p9
    * GLO_L2OF_L2OC_real.bin
       -- for GLONASS L2OF
       -- dataType: int8
       -- fileType: 1（实数采样）
       -- IF: -14e6 Hz
       -- samplingFreq: 70e6 Hz
    
       -- for GLONASS L2OC
       -- dataType: int8
       -- fileType: 1（实数采样）
       -- IF: -11.94e6 Hz
       -- samplingFreq: 70e6 Hz
       -- link: https://pan.baidu.com/s/1P-xWe-aAUu3esymnbsPd-w?pwd=pnr8 password: pnr8
    * L2C_real.bin
       -- for GPS L2C (L2CM and L2CL)
       -- dataType: int8
       -- fileType: 1（实数采样）
       -- IF: -7.4e6 Hz
       -- samplingFreq: 53e6 Hz
       -- link: https://pan.baidu.com/s/16OZPphh6SAObSKf1t2ykPA?pwd=ss7k password: ss7k
    * E5b_B2b_GLO_L3OC_real.bin
       -- for Galileo E5b, BDS B2b
       -- dataType: int8
       -- fileType: 1（实数采样）
       -- IF: +17.14e6 Hz
       -- samplingFreq: 70e6 Hz
    
       -- for GLONASS L3OC
       -- dataType: int8
       -- fileType: 1（实数采样）
       -- IF: +12.025e6 Hz
       -- samplingFreq: 70e6 Hz
       -- link: https://pan.baidu.com/s/1Z-LK5UZ_DQgQk654cY0LgA?pwd=puzw password: puzw
    * L5_E5a_B2a_real.bin
       -- for GPS L5, Galileo E5a and BDS B2a
       -- dataType: int8
       -- fileType: 1（实数采样）
       -- IF: -13.55e6 Hz
       -- samplingFreq: 99.375e6 Hz
       -- link: https://pan.baidu.com/s/1xlftENWf76QwEVzzmxSUDw?pwd=m74w password: m74w
    * B3I_E6_real.bin
       -- for BDS B3I
       -- dataType: int8
       -- fileType: 1（实数采样）
       -- IF: +8.52e6 Hz
       -- samplingFreq: 70e6 Hz
    
       -- for Galileo E6B
       -- dataType: int8
       -- fileType: 1（实数采样）
       -- IF: +18.75e6 Hz
       -- samplingFreq: 70e6 Hz
       -- link: https://pan.baidu.com/s/14q3K1MKNnAtxz64mUu9DKw?pwd=unn8 password: unn8
