import mxnet as mx 
import numpy as np
import os 

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model',default='bert' , type=str)
parser.add_argument('--batchsize',default=1 , type=int)
parser.add_argument('--seq_length',default=128 , type=int)

args = parser.parse_args()

model_name = args.model
batch = args.batchsize
seq_length = args.seq_length

sym = './bert-symbol.json'
params = './bert-0001.params'

onnx_file = "../onnx/bert_mxnet.onnx"
try:
    target_path = "../onnx"
    os.mkdir(target_path)
except:
    pass
batch = 1
seq_length = 128
dtype = "float32"

# shape_dict = {
#     'data0': (batch, seq_length),
#     'data1': (batch, seq_length),
#     'data2': (batch,)
# }

input_shape=[(batch,seq_length),(batch,seq_length),(batch,)]

converted_model_path = mx.onnx.export_model(sym, params, input_shape, np.float32 , onnx_file)
