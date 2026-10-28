An Open-Source Python GNSS SDR Toolbox with SIMD/GPU Acceleration： pySoftGNSS 
===============================================================================



Overview
-------------------------------------------------------------------------------
pySoftGNSS is an open-source Python-based toolbox for post-processing recorded
GNSS intermediate-frequency (IF) signals. It uses the same receiver
processing architecture as mSoftGNSS and organizes signal-processing
functions into Python modules and classes. Supporting GPS, Galileo, GLONASS,
and BeiDou signals, it provides signal-dependent acquisition, data/pilot
tracking, navigation-message decoding, pseudorange generation, and positioning.
CPU and GPU acquisition use NumPy/SciPy and CuPy, respectively. Both
channel-serial and channel-parallel tracking support NumPy-based,
SIMD-accelerated, and GPU-accelerated correlators. SIMD implementations include
x86 AVX2 and ARM64 NEON, selected automatically for the running platform.
Channel-parallel tracking reuses shared IF data buffers and batches correlation 
tasks across channels. C++ and CUDA C++ correlators are compiled into shared 
libraries and accessed through ctypes.CDLL, while scheduling, tracking-loop control,
navigation decoding, and positioning remain in Python. This design preserves
readable, configurable high-level processing while accelerating computationally
intensive sample-level operations. With modernized-signal support, LDPC
decoding, and result visualization, pySoftGNSS supports GNSS algorithm
development, receiver prototyping, and reproducible evaluation.



Authors
-------------------------------------------------------------------------------
* Yafeng Li
    * E-Mail: <lyf8118@126.com>
    * Wechat: lyf8118521
    * QQ Group for Technical Discussions on GNSS Software Receivers: 147304049


* Dennis Akos  
    * E-Mail: <dma@colorado.edu>
    * HP: <http://www.colorado.edu/aerospace/dennis-akos>



Features
-------------------------------------------------------------------------------
* GNSS signal processing functions written in Python
    * Local code generation
    * CPU and GPU acquisition
    * Channel-serial and channel-parallel tracking
    * Python/NumPy, SIMD shared-library, and GPU/CUDA shared-library correlators
    * Data/pilot tracking for modernized signals
    * Carrier-to-noise-density ratio (C/N0) estimation and tracking plots
    * Navigation-message decoding (including LDPC decoder)
    * Pseudorange generation
    * Receiver position/clock-bias calculation and result plotting
* SIMD/GPU acceleration and correlator options
    * settings.gpuACQflag = 0/1 selects CPU/GPU acquisition.
    * settings.trkMode = 0/1 selects channel-serial/channel-parallel tracking.
    * settings.correlatorType = 0 uses Python/NumPy-based correlators.
    * settings.correlatorType = 1 uses CPU SIMD shared-library correlators.
    * settings.correlatorType = 2 uses CUDA GPU shared-library correlators.
    * Python/NumPy-based correlators and signal-specific local-code
      generators are implemented in each receiver's correlator.py.
    * Shared-library correlator backends include BPSK, QPSK, and QMBOC
      SIMD and CUDA variants, called through ctypes.CDLL.
* Supported signals
    * GPS L1 C/A (data)
    * GPS L1C (data + pilot)
    * GPS L2C (data + pilot)
    * GPS L5 (data + pilot)
    * Galileo E1 (data + pilot)
    * Galileo E5a (data + pilot)
    * Galileo E5b (data + pilot)
    * Galileo E6 (data + pilot; HAS page decoding)
    * GLONASS L1OF (data)
    * GLONASS L2OF (data)
    * GLONASS L1OC (data + pilot)
    * GLONASS L2OCp (pilot)
    * GLONASS L3OC (data + pilot)
    * BeiDou B1I (data)
    * BeiDou B3I (data)
    * BDS-3 B1C (data + pilot)
    * BDS-3 B2a (data + pilot)
    * BeiDou B2b (data)
* IF binary-file post-processing
    * settings.dataType selects int8 or int16. Both types support real and
      complex I/Q samples.
    * settings.fileType = 1 selects real samples; 2 selects interleaved
      I/Q samples (I0,Q0,I1,Q1,...). dataType applies to each I/Q component.
    * LimeSDR complex IF and NUT4NT real-sample examples are documented below;
      set the data format, IF, and sampling rate to match the recording.
    * Accelerated correlators are compiled for the target CPU/OS and loaded
      through ctypes without requiring MATLAB.
	


Directory and Files
-------------------------------------------------------------------------------
Root folders

    ./Doc                     Documentation, ICD material, papers, and receiver
                              summary documents.
    ./IF_Data_Set             Optional location for IF data files and metadata.
    ./native_Correlators      Shared SIMD and CUDA correlator source files
                              and locally compiled shared libraries.

Receiver folders

    Each receiver folder is self-contained except for the shared ./native_Correlators,
    ./IF_Data_Set, and ./Doc folders. The current receiver folders are:

    ./pyGPS_L1CA              GPS L1 C/A SDR receiver
    ./pyGPS_L1C               GPS L1C SDR receiver
    ./pyGPS_L2C               GPS L2C SDR receiver
    ./pyGPS_L5                GPS L5 SDR receiver
    ./pyGalileo_E1            Galileo E1 SDR receiver
    ./pyGalileo_E5a           Galileo E5a SDR receiver
    ./pyGalileo_E5b           Galileo E5b SDR receiver
    ./pyGalileo_E6            Galileo E6B/E6C SDR receiver (HAS reception)
    ./pyGlonass_L1_L2         GLONASS L1OF/L2OF SDR receiver
    ./pyGlonass_L1OC          GLONASS L1OC SDR receiver
    ./pyGlonass_L2OC          GLONASS L2OCp SDR receiver
    ./pyGlonass_L3OC          GLONASS L3OC SDR receiver
    ./pyBDS_B1I               BeiDou B1I SDR receiver
    ./pyBDS_B3I               BeiDou B3I SDR receiver
    ./pyBDS-3_B1C             BDS-3 B1C SDR receiver
    ./pyBDS-3_B2a             BDS-3 B2a SDR receiver
    ./pyBDS_B2b               BeiDou B2b SDR receiver

Common layout inside each receiver folder

    ./main.py                 Receiver startup script. It selects the receiver
                              directory, loads settings, plots IF waveforms and
                              spectra, and asks whether processing should start.
    ./initSettings.py         Settings class for receiver configuration,
                              probeData(), and top-level postProcessing().
    ./acquisition.py          AcqEngine class with CPU and CuPy GPU acquisition,
                              result preparation, and acquisition plotting.
    ./tracking.py             Channel and TrackingEngine classes with
                              channel-serial/channel-parallel tracking and plotting.
    ./postNavigation.py       Ephemeris, NavSolutions, and NavigationEngine classes
                              for navigation decoding and position computation
                              where implemented. 
    ./correlator.py           Local-code generation/sampling, Python/NumPy
                              correlators, and ctypes shared-library wrapper classes.
    ./channelDecoder.py       LDPC decoder classes and associated BCH decoders;
                              present only in pyBDS-3_B1C, pyBDS-3_B2a,
                              pyBDS_B2b, and pyGPS_L1C.
    ./commUtils.py            Receiver-local common math and positioning utilities:
                              pseudoranges, coordinate transforms, least-squares
                              positioning, loop coefficients, and progress display.
    ./result_Cache            Acquisition, tracking, and navigation/ephemeris
                              results saved as .pkl files. Keep this directory.
                              Galileo E1 also requires E1b.dat/E1c.dat here;
                              E6B requires E6B_codes.dat/E6C_codes.dat here.
    ./assets                  Galileo E6B HAS Reed-Solomon generator matrix;
                              present in pyGalileo_E6.

Shared correlator backends in ./native_Correlators

    corrX86SIMDSerial*.cpp      x86 AVX2 serial tracking correlator sources.
    corrX86SIMDParallel*.cpp    x86 AVX2 channel-parallel correlator sources.
    corrNeonSIMD*.cpp           ARM64 NEON serial/channel-parallel correlator sources.
    corrGPUSerial*.cu           CUDA serial tracking correlator source files.
    corrGPUParallel*.cu         CUDA channel-parallel tracking correlator source files.
    corrGPUParallelFused*.cu    CUDA fused channel-parallel tracking correlator
                                source files where available.
    *.dll / *.so                Locally compiled libraries selected by settings.correlatorType.

Correlator selection

    settings.correlatorType = 0
                              Python/NumPy-based correlators in the receiver's
                              correlator.py. These are easiest to inspect
                              and debug.
    settings.correlatorType = 1
                              CPU SIMD shared-library correlators from ./native_Correlators.
                              The code automatically selects x86 or ARM64 NEON
                              according to the running platform architecture.
                              These accelerate tracking without requiring CUDA.
    settings.correlatorType = 2
                              CUDA GPU shared-library correlators from ./native_Correlators.
                              These are intended for high-throughput serial or
                              channel-parallel tracking.



Software Dependencies
-------------------------------------------------------------------------------
* Python 3.10 or later with NumPy, SciPy, and Matplotlib is required.
  Native correlators require matching 64-bit Python and library architectures.
* SciPy provides signal-processing and numerical utilities; Matplotlib displays
  IF spectra and receiver results.
* Navigation decoders are implemented in postNavigation.py and, for the four
  LDPC-based receivers, channelDecoder.py. MATLAB toolboxes are not required.
* CuPy with a compatible CUDA installation and NVIDIA GPU/driver is needed
  for GPU acquisition. Native CUDA shared-library correlators require a compatible
  NVIDIA GPU and driver; CuPy is not needed for the shared-library calls themselves.
* x86 SIMD correlators require AVX2 and FMA; ARM64 uses the NEON sources.
  SIMD builds need a C++ compiler; CUDA builds require nvcc and a supported
  host compiler. Place the compiled .dll and .so files directly in
  native_Correlators.
* Tkinter is used for tracking progress windows and should be available in
  the Python environment.
* CPU-only operation is possible by selecting Python/NumPy or SIMD correlators
  and disabling GPU acquisition in each receiver's initSettings.py.
  
  
 
Compile correlators
-------------------------------------------------------------------------------
This section describes how to compile the native correlators for pySoftGNSS.
Choose the instructions for your operating system and CPU architecture.

Run commands in a system terminal, starting in the pySoftGNSS root, and follow
only the section for your platform. Make the required compilers available on
PATH. Exit Python processes using the libraries before rebuilding them.

| Platform | SIMD requirements | Output files |
| --- | --- | --- |
| Windows x86-64 | x86-64 CPU with AVX2 and FMA support | .dll |
| Linux x86-64 | x86-64 CPU with AVX2 and FMA support | .so |
| Linux ARM64 | AArch64 CPU with NEON support | .so |

Enter native_Correlators and select the SIMD and CUDA sources for the target
architecture. Python calls their C interfaces through ctypes.

SIMD compilation requires only a C++ compiler for the target platform. CUDA
compilation requires the CUDA Toolkit and a host compiler supported by nvcc.
Running the GPU correlators also requires a compatible NVIDIA GPU and driver.

The cudaArch / CUDA_ARCH variables below specify the target GPU's compute
capability. Windows and Linux x86-64 use '89' for an RTX 4080 Laptop GPU as
an example.

Linux ARM64 uses '87' for a Jetson Orin NX as an example. For another GPU,
change this value to its actual compute capability and ensure that the
selected nvcc supports the target.

### Windows x86-64

The SIMD example uses 64-bit MinGW-w64/G++. Run the commands in PowerShell.
"g++ -dumpmachine" should identify an x86_64 Windows target. Enable both
AVX2 and FMA.

```powershell
Set-Location .\native_Correlators
Get-ChildItem -File 'corrX86SIMD*.cpp' | ForEach-Object {
    & g++ -O3 -std=c++14 -mavx2 -mfma -shared -static-libgcc -static-libstdc++ $_.FullName -o "$($_.BaseName).dll"
    if ($LASTEXITCODE -ne 0) { throw "Build failed: $($_.Name)" }
}
```

CUDA compilation uses nvcc and Microsoft Visual C++. Initialize a compatible
x64 MSVC environment first, for example by starting PowerShell from Visual
Studio's "x64 Native Tools Command Prompt". Enter the same native_Correlators
directory, then run the commands below with cudaArch set for the GPU.

```powershell
$cudaArch = '89'
Get-ChildItem -File 'corrGPU*.cu' | ForEach-Object {
    & nvcc -O3 -std=c++14 --shared --cudart static -Xcompiler /MD -Xcompiler /utf-8 "-gencode=arch=compute_$cudaArch,code=sm_$cudaArch" $_.FullName -o "$($_.BaseName).dll"
    if ($LASTEXITCODE -ne 0) { throw "Build failed: $($_.Name)" }
}
```

After compiling both the SIMD and CUDA correlators, the .dll files are
generated in this directory. The CUDA runtime is linked statically, but the
NVIDIA driver and Microsoft C++ runtime are still required.

### Linux x86-64

Use native x86-64 GCC/G++ to compile SIMD in Bash:

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

CUDA compilation uses nvcc and a GCC/G++ version it supports. Both should be
available on PATH. Run the following in the same native_Correlators directory,
with CUDA_ARCH set for the actual GPU:

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

After compilation, the x86 SIMD .so and CUDA .so files are generated in this
directory.

### Linux ARM64

Use native ARM64 Python and AArch64 GCC/G++. Run the commands in Bash on the
ARM64 machine. SIMD uses the separate NEON sources.

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

Compiling the CUDA correlators requires the corresponding ARM64 CUDA toolchain;
running them also requires a compatible NVIDIA GPU and driver. Run the following in the same
native_Correlators directory, with CUDA_ARCH set for the actual GPU. Keep
-ffp-contract=off and --fmad=false to preserve the numerical compilation
settings of this ARM implementation.

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

After compiling both the SIMD and CUDA correlators, the NEON .so and CUDA .so
files are generated in this directory.



How to use
-------------------------------------------------------------------------------
* Step 1: Enter the receiver folder for the target signal, for example
          "pyBDS-3_B1C" or "pyGPS_L1CA".
* Step 2: Put the IF data file under "IF_Data_Set", or set an absolute file
          path in settings.fileName.
* Step 3: Configure processing, IF-data, acquisition, tracking, correlator, and
          navigation parameters in "initSettings.py". skipNumberOfSamples counts
          logical samples: one real value or one I/Q pair counts as one sample.
* Step 4: Select CPU/GPU acquisition with settings.gpuACQflag, channel-serial or
          channel-parallel tracking with settings.trkMode, and script-based/SIMD/GPU
          correlation with settings.correlatorType.
* Step 5: Run "python main.py" from the receiver folder. Inspect the IF-data plots,
          then enter 1 to start processing or 0 to exit.
* Step 6: After processing, review the acquisition, tracking, navigation, and
          positioning results and their plots.



Implementation details
-------------------------------------------------------------------------------
The receiver set follows the SoftGNSS processing architecture and organizes
its Python implementation into modules and classes. It supports modern GNSS
signals, CPU/GPU acquisition, channel-serial and channel-parallel tracking,
and shared SIMD/CUDA correlator libraries called through ctypes.CDLL.

Channel-parallel tracking uses an outer loop over tracking epochs, shared IF
buffering, and batched correlator calls to process all active channels before
updating their individual tracking loops. This architecture preserves independent
channel states and loop feedback, reduces redundant IF reads and interface-call
overhead, and supports multichannel batch processing with SIMD/GPU acceleration.

See the reference for the BDS-3 B1C/B2a receiver design:
Li, Y., Shivaramaiah, N.C. & Akos, D.M. Design and implementation of an open-source 
BDS-3 B1C/B2a SDR receiver. GPS Solut 23, 60 (2019). 
https://doi.org/10.1007/s10291-019-0853-z


	   
Test signal (collected by LimeSDR) and parameter configurations
-------------------------------------------------------------------------------
    * L1CA_E1_B1C.bin
       -- for GPS L1 C/A, Galileo E1 and BDS B1C 
       -- dataType: int16
       -- fileType: 2 (complex I/Q samples; each component is int16)
       -- IF: 0 Hz
       -- samplingFreq: 30e6 Hz
       -- link: https://pan.baidu.com/s/1NdZeVrEdEjDnJiV91Cg0qg?pwd=bvh5 password: bvh5
    * GPS L1C.iq
       -- for GPS L1C 
       -- dataType: int8
       -- fileType: 2 (complex I/Q samples; each component is int8)
       -- IF: 0 Hz
       -- samplingFreq: 25e6 Hz
       -- link: https://pan.baidu.com/s/1hZ4S8R4VWB5OtPr9VyPCYQ?pwd=tv8b password: tv8b
    * GPS_L2.bin
       -- for GPS L2C
       -- dataType: int16
       -- fileType: 2 (complex I/Q samples; each component is int16)
       -- IF: 0 Hz
       -- samplingFreq: 30e6 Hz
       -- link: https://pan.baidu.com/s/1KqdyWyuVPAgx0EoUdoLwag?pwd=5tya password: 5tya
    * L5_E5a_B2a.bin
       -- for GPS L5, Galileo E5a and BDS B2a
       -- dataType: int16
       -- fileType: 2 (complex I/Q samples; each component is int16)
       -- IF: 0 Hz
       -- samplingFreq: 30e6 Hz
       -- link: https://pan.baidu.com/s/1rX2lguJgMmgjAdZIJK1FNg?pwd=bnzm password: bnzm
    
    * E5b_B2b.bin   
       -- for Galileo E5b, BDS B2b
       -- dataType: int16
       -- fileType: 2 (complex I/Q samples; each component is int16)
       -- IF: 0 Hz
       -- samplingFreq: 30e6 Hz
       -- link: https://pan.baidu.com/s/128dwHIKUG9iM0MigS_7vJw?pwd=x4uq password: x4uq
    * GAL_E6.bin  
       -- for Galileo E6
       -- dataType: int16
       -- fileType: 2 (complex I/Q samples; each component is int16)
       -- IF: 0 Hz
       -- samplingFreq: 30e6 Hz
       -- link: https://pan.baidu.com/s/1OrHtpNG-dQxSmmVc9rX6tA?pwd=qfx8 password: qfx8
    * GLO_L1OC_L1.bin 
       -- for GLONASS L1OF  
       -- dataType: int16
       -- fileType: 2 (complex I/Q samples; each component is int16)
       -- IF: 1.005e6 Hz
       -- samplingFreq: 30e6 Hz
       -- GLONASS L1OC
       -- dataType: int16
       -- fileType: 2 (complex I/Q samples; each component is int16)
       -- IF: 0 Hz
       -- samplingFreq: 30e6 Hz
       -- link: https://pan.baidu.com/s/1w-fYtmM3oW_pxR4ZH_HGyw?pwd=js93 password: js93
    * GLO_L2OC_L2.bin  
       -- for GLONASS L2OF 
       -- dataType: int16
       -- fileType: 2 (complex I/Q samples; each component is int16)
       -- IF: -2.06e6 Hz
       -- samplingFreq: 30e6 Hz
    
       -- for GLONASS L2OC 
       -- dataType: int16
       -- fileType: 2 (complex I/Q samples; each component is int16)
       -- IF: 0 Hz
       -- samplingFreq: 30e6 Hz
       -- link: https://pan.baidu.com/s/1jefXiCkeI9qHSqwZ_RXr2g?pwd=33w6 password: 33w6
    * GLO_L3OC.bin  
       -- for GLONASS L3OC	
       -- dataType: int16
       -- fileType: 2 (complex I/Q samples; each component is int16)
       -- IF: 0 Hz
       -- samplingFreq: 30e6 Hz
       -- link: https://pan.baidu.com/s/1ySNHn3W66Q6n9fxERR7nOQ?pwd=289m password: 289m
    * BDS_B1I.bin
       -- for BDS B1I 
       -- dataType: int16
       -- fileType: 2 (complex I/Q samples; each component is int16)
       -- IF: 0 Hz
       -- samplingFreq: 30e6 Hz
       -- link: https://pan.baidu.com/s/1qV-TCtROGHL8iJZlqw5mZA?pwd=39vd password: 39vd
    * BDS_B3I.bin       
       -- for BDS B3I    
       -- dataType: int16
       -- fileType: 2 (complex I/Q samples; each component is int16)
       -- IF: 0 Hz
       -- samplingFreq: 30e6 Hz
       -- link: https://pan.baidu.com/s/1T_g1to1e8OW5OFtHlQsZeg?pwd=6w1u password: 6w1u 



Test signal (collected by NUT4NT) and parameter configurations
-------------------------------------------------------------------------------
    * L1CA_E1_B1C_real.bin
       -- for GPS L1 C/A, Galileo E1 and BDS B1C
       -- dataType: int8
       -- fileType: 1 (real samples)
       -- IF: -14.58e6 Hz
       -- samplingFreq: 53e6 Hz
       -- link: https://pan.baidu.com/s/1cFGCBwcWA0vHBeiL7WU_2Q?pwd=r8au password: r8au
    * B1I_real.bin
       -- for BDS B1I
       -- dataType: int8
       -- fileType: 1 (real samples)
       -- IF: -28.902e6 Hz
       -- samplingFreq: 99.375e6 Hz
       -- link: https://pan.baidu.com/s/1y4LfFZSO-Qm3Tvshvg_Gzg?pwd=86ii password: 86ii
    * GLO_L1OF_L1OC_real.bin
       -- for GLONASS L1OF
       -- dataType: int8
       -- fileType: 1 (real samples)
       -- IF: +12e6 Hz
       -- samplingFreq: 53e6 Hz
    
       -- for GLONASS L1OC
       -- dataType: int8
       -- fileType: 1 (real samples)
       -- IF: +10.995e6 Hz
       -- samplingFreq: 53e6 Hz
       -- link: https://pan.baidu.com/s/1-fxXe6VW36izl2ve3dMQsw?pwd=j7p9 password: j7p9
    * GLO_L2OF_L2OC_real.bin
       -- for GLONASS L2OF
       -- dataType: int8
       -- fileType: 1 (real samples)
       -- IF: -14e6 Hz
       -- samplingFreq: 70e6 Hz
    
       -- for GLONASS L2OC
       -- dataType: int8
       -- fileType: 1 (real samples)
       -- IF: -11.94e6 Hz
       -- samplingFreq: 70e6 Hz
       -- link: https://pan.baidu.com/s/1P-xWe-aAUu3esymnbsPd-w?pwd=pnr8 password: pnr8
    * L2C_real.bin
       -- for GPS L2C (L2CM and L2CL)
       -- dataType: int8
       -- fileType: 1 (real samples)
       -- IF: -7.4e6 Hz
       -- samplingFreq: 53e6 Hz
       -- link: https://pan.baidu.com/s/16OZPphh6SAObSKf1t2ykPA?pwd=ss7k password: ss7k
    * E5b_B2b_GLO_L3OC_real.bin
       -- for Galileo E5b, BDS B2b
       -- dataType: int8
       -- fileType: 1 (real samples)
       -- IF: +17.14e6 Hz
       -- samplingFreq: 70e6 Hz
    
       -- for GLONASS L3OC
       -- dataType: int8
       -- fileType: 1 (real samples)
       -- IF: +12.025e6 Hz
       -- samplingFreq: 70e6 Hz
       -- link: https://pan.baidu.com/s/1Z-LK5UZ_DQgQk654cY0LgA?pwd=puzw password: puzw
    * L5_E5a_B2a_real.bin
       -- for GPS L5, Galileo E5a and BDS B2a
       -- dataType: int8
       -- fileType: 1 (real samples)
       -- IF: -13.55e6 Hz
       -- samplingFreq: 99.375e6 Hz
       -- link: https://pan.baidu.com/s/1xlftENWf76QwEVzzmxSUDw?pwd=m74w password: m74w
    * B3I_E6_real.bin
       -- for BDS B3I
       -- dataType: int8
       -- fileType: 1 (real samples)
       -- IF: +8.52e6 Hz
       -- samplingFreq: 70e6 Hz
    
       -- for Galileo E6B
       -- dataType: int8
       -- fileType: 1 (real samples)
       -- IF: +18.75e6 Hz
       -- samplingFreq: 70e6 Hz
       -- link: https://pan.baidu.com/s/14q3K1MKNnAtxz64mUu9DKw?pwd=unn8 password: unn8
