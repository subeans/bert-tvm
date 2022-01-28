import numpy as np
import onnxruntime as ort
import argparse
import time
import warnings

warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--model',default='bert' , type=str)
parser.add_argument('--model_type',default='nlp' , type=str)
parser.add_argument('--framework',default='mxnet' , type=str)
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
dtype = "float32"
inputs = np.random.randint(0, 2000, size=(batch, seq_length)).astype(dtype)
token_types = np.random.uniform(size=(batch, seq_length)).astype(dtype)
valid_length = np.asarray([seq_length] * batch).astype(dtype)

# Convert to MXNet NDArray and run the MXNet model

shape_dict = {
    'data0': (batch, seq_length),
    'data1': (batch, seq_length),
    'data2': (batch,)
}

model_path = f"./{model_name}_{framework}.onnx"


session = ort.InferenceSession(model_path)
session.get_modelmeta()
inname = [input.name for input in session.get_inputs()]
outname = [output.name for output in session.get_outputs()]
    
    
time_list = []
for i in range(count):
    start_time = time.time()
    session.run(outname, {inname[0]: inputs,inname[1]:token_types,inname[2]:valid_length})
    running_time = time.time() - start_time
    print(f"ONNX serving {model_name}-{batch} inference latency : ",(running_time)*1000,"ms")
    time_list.append(running_time)

time_medium = np.median(np.array(time_list))
print('{} median latency (batch={}): {} ms'.format(model_name, batch, time_medium * 1000))
print('{} mean latency (batch={}): {} ms'.format(model_name, batch, np.mean(np.array(time_list)*1000)))
