@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "PROFILE=full"
set "DO_CLEAN=0"
for %%A in (%*) do (
    if /I "%%~A"=="--clean" (
        set "DO_CLEAN=1"
    ) else (
        set "PROFILE=%%~A"
    )
)

set "DO_FULL=0"
set "DO_SM89=0"
set "DO_CPU=1"
set "DO_CPU_COMPAT=1"
set "DO_VULKAN=0"
if /I "%PROFILE%"=="full" (
    set "DO_FULL=1"
) else if /I "%PROFILE%"=="cuda-full" (
    set "DO_FULL=1"
    set "DO_CPU=0"
    set "DO_CPU_COMPAT=0"
) else if /I "%PROFILE%"=="sm89" (
    set "DO_SM89=1"
) else if /I "%PROFILE%"=="cuda-sm89" (
    set "DO_SM89=1"
    set "DO_CPU=0"
    set "DO_CPU_COMPAT=0"
) else if /I "%PROFILE%"=="all" (
    set "DO_FULL=1"
    set "DO_SM89=1"
) else if /I "%PROFILE%"=="vulkan" (
    set "DO_VULKAN=1"
    set "DO_CPU=0"
    set "DO_CPU_COMPAT=0"
) else if /I "%PROFILE%"=="cpu" (
    set "DO_CPU=1"
) else (
    echo Unknown profile "%PROFILE%". Usage: build-llama.bat [full^|sm89^|all^|cpu^|vulkan] [--clean]
    exit /b 1
)
if defined VULKAN_SDK (
    if exist "%VULKAN_SDK%\Include\vulkan\vulkan.h" set "DO_VULKAN=1"
)

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" goto NOVS

echo === USING %VCVARS% ===
call "%VCVARS%"
if errorlevel 1 goto FAILED

where ninja >nul 2>&1
if errorlevel 1 set "PATH=%PATH%;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
where ninja >nul 2>&1
if errorlevel 1 goto NONINJA

cd llamacpp

set "COMMON_FLAGS=-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_CONNECT=OFF"
set "CPU_AVX2_FLAGS=-DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON"
set "CPU_COMPAT_FLAGS=-DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_SSE42=ON"
set "TARGETS=llama-server llama-bench llama-perplexity"

if "%DO_FULL%"=="1" (
    echo === CONFIGURE CUDA ^(full: 75;80;86;89;120-virtual^) ===
    if "%DO_CLEAN%"=="1" if exist build-cuda rmdir /s /q build-cuda
    cmake -B build-cuda %COMMON_FLAGS% %CPU_AVX2_FLAGS% -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;120-virtual" -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/nvcc.exe" -DGGML_CUDA_FA_ALL_QUANTS=ON
    if errorlevel 1 goto CUDAFAIL

    echo === BUILD CUDA ^(full^) ===
    cmake --build build-cuda --config Release --target %TARGETS% -j 12
    if errorlevel 1 goto CUDAFAIL
)

if "%DO_SM89%"=="1" (
    echo === CONFIGURE CUDA ^(sm89: 89^) ===
    if "%DO_CLEAN%"=="1" if exist build-cuda-sm89 rmdir /s /q build-cuda-sm89
    cmake -B build-cuda-sm89 %COMMON_FLAGS% %CPU_AVX2_FLAGS% -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="89" -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/nvcc.exe" -DGGML_CUDA_FA_ALL_QUANTS=ON
    if errorlevel 1 goto CUDAFAIL

    echo === BUILD CUDA ^(sm89^) ===
    cmake --build build-cuda-sm89 --config Release --target %TARGETS% -j 12
    if errorlevel 1 goto CUDAFAIL
)

if "%DO_VULKAN%"=="1" (
    echo === CONFIGURE VULKAN ===
    if "%DO_CLEAN%"=="1" if exist build-vulkan rmdir /s /q build-vulkan
    cmake -B build-vulkan %COMMON_FLAGS% %CPU_COMPAT_FLAGS% -DGGML_VULKAN=ON
    if errorlevel 1 goto CPUFAIL

    echo === BUILD VULKAN ===
    cmake --build build-vulkan --config Release --target %TARGETS% -j 12
    if errorlevel 1 goto CPUFAIL
)

if "%DO_CPU%"=="1" (
    echo === CONFIGURE CPU ^(AVX2^) ===
    if "%DO_CLEAN%"=="1" if exist build-cpu rmdir /s /q build-cpu
    cmake -B build-cpu %COMMON_FLAGS% %CPU_AVX2_FLAGS%
    if errorlevel 1 goto CPUFAIL

    echo === BUILD CPU ^(AVX2^) ===
    cmake --build build-cpu --config Release --target %TARGETS% -j 12
    if errorlevel 1 goto CPUFAIL
)

if "%DO_CPU_COMPAT%"=="1" (
    echo === CONFIGURE CPU ^(SSE4.2 compat^) ===
    if "%DO_CLEAN%"=="1" if exist build-cpu-compat rmdir /s /q build-cpu-compat
    cmake -B build-cpu-compat %COMMON_FLAGS% %CPU_COMPAT_FLAGS%
    if errorlevel 1 goto CPUFAIL

    echo === BUILD CPU ^(SSE4.2 compat^) ===
    cmake --build build-cpu-compat --config Release --target %TARGETS% -j 12
    if errorlevel 1 goto CPUFAIL
)

echo === ALL DONE ===
goto :eof

:NOVS
echo NO_VISUAL_STUDIO_FOUND
exit /b 1
:NONINJA
echo NO_NINJA
exit /b 1
:CUDAFAIL
echo CUDA_FAILED
exit /b 1
:CPUFAIL
echo CPU_FAILED
exit /b 1
:FAILED
echo VCVARS_FAILED
exit /b 1
