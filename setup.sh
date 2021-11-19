cd det3d/ops/dcn 
python setup.py build_ext --inplace

cd .. && cd  iou3d_nms
python setup.py build_ext --inplace

cd .. && cd  primitives
python setup.py build_ext --inplace

cd .. && cd .. && cd ../PytorchHashmap/torch_hash
python setup.py build_ext --inplace

