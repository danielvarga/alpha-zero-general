# Get tensorflow C-API
wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.15.0.tar.gz
mkdir tflib
tar -C tflib -xzf libtensorflow-gpu-linux-x86_64-1.15.0.tar.gz
export LIBRARY_PATH=$LIBRARY_PATH:./tflib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./tflib

act = pwd
# Compile with:
# gcc -I/$act/tflib/include -L/$act/tflib/lib  main.cpp -ltensorflow -o main

# Get CPP-FLOW
git clone https://github.com/serizba/cppflow.git
