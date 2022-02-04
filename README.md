# bert-tvm
Bert Inference Serving with ONNX and TVM 

## Framework 
### MXNET 
- Running from source 
```
git clone https://github.com/subeans/bert-tvm.git
cd bert-tvm/mxnet 

# 1. base line - mxnet serving 
python3 mxnet_serving.py # mxnet serving and export mxnet model(params, json)

# 2. optimize with onnx - onnx serving 
python3 mx2onnx.py # mxnet model to onnx 
cd ../onnx/ 
python3 onnx_serving.py # onnx serving 

# 3. optimize with tvm - tvm serving 
cd ../mxnet 
python3 mxnet_tvm_serving.py

```
