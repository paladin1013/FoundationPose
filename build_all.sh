DIR=$(pwd)

mamba activate foundation-pose
cd $DIR/mycpp/ && mkdir -p build && cd build && cmake .. && make -j$(($(nproc)-1))
cd ~/kaolin && rm -rf build *egg* && pip install -e .
cd $DIR/bundlesdf/mycuda && rm -rf build *egg* && pip install -e .

cd ${DIR}