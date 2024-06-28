from vllm import LLM, SamplingParams
import torch
import json
from transformers import AutoTokenizer
import pdb
from itertools import groupby

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", gpu_memory_utilization=0.5,
          #tokenizer=tokenizer,
          )
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
llm.set_tokenizer(tokenizer)

pdb.set_trace()

cache_driver = llm.llm_engine.model_executor.driver_worker.model_runner.lmcache_driver
#for sample_idx in range(3,6):
sample_idx = 3
f = open(f"inputs/{sample_idx}.json")
ex = json.load(f)
chunk_num = ex['chunk_num']
doc_prompts = [ex[f'{i}'] for i in range(chunk_num)]
q_prompt = ex['query']
doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
q_ids = tokenizer.encode(q_prompt)[1:]


# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, max_tokens=1)


cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata
cache_fuse_metadata['collect'] = False
cache_fuse_metadata['check'] = False

s_start_full = [733, 4138, 28793]
s_start_len = len(s_start_full) + 1

#s_start = [518, 25580, 29962]
s_start = []
s_start_1_len = len(s_start) + 1

s_end = [733, 28748, 16289, 28793]
s_end_len = len(s_end)
old_kvs = []

doc_chunk_ids = [chunk_ids for chunk_ids in doc_chunk_ids]
doc_chunk_ids = [s_start_full] + doc_chunk_ids
doc_chunk_ids = doc_chunk_ids + [q_ids+s_end]

doc_chunk_ids_with_sep = []
for idx, chunk_ids in enumerate(doc_chunk_ids):
    if idx == 0:
        doc_chunk_ids_with_sep.append(chunk_ids)
    else:
        doc_chunk_ids_with_sep.append([422, 422]+chunk_ids)
input_ids = []
for i in range(len(doc_chunk_ids)):
    temp_ids = doc_chunk_ids_with_sep[i]
    input_ids += temp_ids
print(len(input_ids))
pdb.set_trace()
input_prompt = tokenizer.decode(input_ids)

sampling_params = SamplingParams(temperature=0, max_tokens=1)

torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
cache_fuse_metadata["check"] = True
cache_fuse_metadata['collect'] = False
output = llm.generate([input_prompt], sampling_params)
end.record()
torch.cuda.synchronize()
partial_time = start.elapsed_time(end)/1000
print(f"Cached generation: {output[0].outputs[0].text}")
print(f"TTFT with cache: {partial_time}")

torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
cache_fuse_metadata["check"] = True
cache_fuse_metadata['collect'] = False
output = llm.generate([input_prompt], sampling_params)
end.record()
torch.cuda.synchronize()
partial_time = start.elapsed_time(end)/1000
print(f"Cached generation: {output[0].outputs[0].text}")
print(f"TTFT with cache: {partial_time}")

torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
cache_fuse_metadata["check"] = True
cache_fuse_metadata['collect'] = False
output = llm.generate([input_prompt], sampling_params)
end.record()
torch.cuda.synchronize()
partial_time = start.elapsed_time(end)/1000
print(f"Cached generation: {output[0].outputs[0].text}")
print(f"TTFT with cache: {partial_time}")