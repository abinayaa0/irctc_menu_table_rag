import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
from FlagEmbedding import BGEM3FlagModel
import time

print("Loading model with use_fp16=False...")
try:
    model2 = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    print("Success loading model2")
    t0 = time.time()
    res = model2.encode(["Test query"], return_dense=True, return_sparse=True)
    print("Success encoding with model2 in", time.time() - t0)
except Exception as e:
    print("Failed model2:", e)
