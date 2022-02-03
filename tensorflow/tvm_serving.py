import numpy as np
import onnxruntime as ort
import onnx
import argparse
import time
import warnings
from tvm import relay
import tvm
import tvm.contrib.graph_runtime as runtime

warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--model',default='bert' , type=str)
parser.add_argument('--model_type',default='nlp' , type=str)
parser.add_argument('--framework',default='tf' , type=str)
parser.add_argument('--batchsize',default=1 , type=int)
parser.add_argument('--seq_length',default=128 , type=int)

args = parser.parse_args()

model_name = args.model
model_type = args.model_type
batch = args.batchsize
seq_length = args.seq_length
framework = args.framework
count = 5 

# Prepare input data
dtype = "int32"
inputs = np.random.randint(0, 2000, size=(batch, seq_length)).astype(dtype)
token_types = np.random.randint(0,1,size=(batch, seq_length)).astype(dtype)
valid_length = np.random.randint(0,1,size=(batch, seq_length)).astype(dtype)

#valid_length = np.asarray([seq_length] * batch).astype(dtype)

# Convert to MXNet NDArray and run the MXNet model

shape_dict = {
    'input_ids': (batch, seq_length),
    'token_type_ids': (batch, seq_length),
    'attention_mask': (batch,)
}

model_path = f"./{model_name}_{framework}.onnx"

onnx_model = onnx.load(model_path)

##### Convert tensorflow model 
print("ONNX model imported to relay frontend.")
mod, params = relay.frontend.from_onnx(onnx_model, shape=shape_dict)


# Compile the imported model
#target = "llvm"
target = "llvm -mcpu=skylake-avx512"
with relay.build_config(opt_level=3, required_pass=["FastMath"]):
    graph, lib, cparams = relay.build(mod, target, params=params)

# Create the executor and set the parameters and inputs
ctx = tvm.cpu()
rt = runtime.create(graph, lib, ctx)
rt.set_input(**cparams)
rt.set_input(data0=inputs, data1=token_types, data2=valid_length)

# Run the executor and validate the correctness
rt.run()
out = rt.get_output(0)

# Benchmark the TVM latency
ftimer = rt.module.time_evaluator("run", ctx, repeat=3, min_repeat_ms=1000)
prof_res = np.array(ftimer().results) * 1000
print(f"TVM latency for batch {batch} and seq length {seq_length}: {np.mean(prof_res):.2f} ms")
