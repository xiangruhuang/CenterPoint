cd det3d/ops/dcn 
rm -rf build/ *.so
python setup.py clean
python setup.py build_ext --inplace

cd .. && cd  iou3d_nms
rm -rf build/ *.so
python setup.py clean
python setup.py build_ext --inplace

cd .. && cd  primitives
rm -rf build/ *.so
python setup.py clean
python setup.py build_ext --inplace

cd .. && cd .. && cd ../PytorchHashmap/torch_hash
rm -rf build/ *.so
python setup.py clean
python setup.py build_ext --inplace

