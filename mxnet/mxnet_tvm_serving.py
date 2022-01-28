import time
import argparse
import numpy as np
import mxnet as mx
import gluonnlp as nlp
import tvm
from tvm import relay
import tvm.contrib.graph_runtime as runtime
import tvm.testing
import warnings

warnings.filterwarnings(action='ignore') 


def timer(thunk, repeat=1, number=10, dryrun=3, min_repeat_ms=1000):
    """Helper function to time a function"""
    for i in range(dryrun):
        thunk()
    ret = []
    for _ in range(repeat):
        while True:
            beg = time.time()
            for _ in range(number):
                thunk()
            end = time.time()
            lat = (end - beg) * 1e3
            if lat >= min_repeat_ms:
                break
            number = int(max(min_repeat_ms / (lat / number) + 1, number * 1.618))
        ret.append(lat / number)
    return ret


parser = argparse.ArgumentParser(description="Optimize BERT-base model from GluonNLP")
parser.add_argument("-b", "--batch", type=int, default=1,
                    help="Batch size (default: 1)")
parser.add_argument("-l", "--length", type=int, default=128,
                    help="Sequence length (default: 128)")
args = parser.parse_args()
batch = args.batch
seq_length = args.length


# Instantiate a BERT classifier using GluonNLP
model_name = 'bert_12_768_12'
dataset = 'book_corpus_wiki_en_uncased'
mx_ctx = mx.cpu()
bert, _ = nlp.model.get_model(
    name=model_name,
    ctx=mx_ctx,
    dataset_name=dataset,
    pretrained=False,
    use_pooler=True,
    use_decoder=False,
    use_classifier=False)
model = nlp.model.BERTClassifier(bert, dropout=0.1, num_classes=2)
model.initialize(ctx=mx_ctx)
model.hybridize(static_alloc=True)

# Prepare input data
dtype = "float32"
inputs = np.random.randint(0, 2000, size=(batch, seq_length)).astype(dtype)
token_types = np.random.uniform(size=(batch, seq_length)).astype(dtype)
valid_length = np.asarray([seq_length] * batch).astype(dtype)

# Convert to MXNet NDArray and run the MXNet model
inputs_nd = mx.nd.array(inputs, ctx=mx_ctx)
token_types_nd = mx.nd.array(token_types, ctx=mx_ctx)
valid_length_nd = mx.nd.array(valid_length, ctx=mx_ctx)
mx_out = model(inputs_nd, token_types_nd, valid_length_nd)
mx_out.wait_to_read()

######################################
# Optimize the BERT model using TVM
######################################

# First, Convert the MXNet model into TVM Relay format
shape_dict = {
    'data0': (batch, seq_length),
    'data1': (batch, seq_length),
    'data2': (batch,)
}
mod, params = relay.frontend.from_mxnet(model, shape_dict)

# Compile the imported model
target = "llvm"
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
tvm.testing.assert_allclose(out.asnumpy(), mx_out.asnumpy(), rtol=1e-3, atol=1e-3)

# Benchmark the TVM latency
ftimer = rt.module.time_evaluator("run", ctx, repeat=3, min_repeat_ms=1000)
prof_res = np.array(ftimer().results) * 1000
print(f"TVM latency for batch {batch} and seq length {seq_length}: {np.mean(prof_res):.2f} ms")
