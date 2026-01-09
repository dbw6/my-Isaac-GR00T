# 获取当前绝对路径
export PROJECT_ROOT=$(pwd)

# 将局部库的路径加入 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PROJECT_ROOT/local_ffmpeg/lib:$LD_LIBRARY_PATH