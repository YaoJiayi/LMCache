from vllm import LLM, SamplingParams
import torch
import json
from transformers import AutoTokenizer
import pdb

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

doc_chunk_ids = [s_start+chunk_ids for chunk_ids in doc_chunk_ids]
doc_chunk_ids = [s_start_full] + doc_chunk_ids
doc_chunk_ids = doc_chunk_ids + [q_ids+s_end]


cache_fuse_metadata['collect'] = True
cache_fuse_metadata["check"] = False

num_layer = 32

input_ids = []
doc_chunk_ids_store = []
for i in range(len(doc_chunk_ids)):
    if i == 0:
        temp_ids = [1] + doc_chunk_ids[i]
    else:
        temp_ids = doc_chunk_ids[i][s_start_1_len-1:]
    input_ids += temp_ids
    doc_chunk_ids_store.append(temp_ids)
    
input_prompt = tokenizer.decode(input_ids)
print(len(input_ids))
pdb.set_trace()
# Concatenate old KVs
for i in range(len(doc_chunk_ids)):
    prompts = [tokenizer.decode(doc_chunk_ids[i])]
    llm.generate(prompts, sampling_params)
    
    # TODO (Jiayi): Move the following part to collect and store
    k_tensor = None
    v_tensor = None
    
    llm_layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
    for j in range(num_layer):
        past_key_values = llm_layers[j].self_attn.hack_kv
        if i == 0:
            temp_k = past_key_values[0][:s_start_len].clone() # do not chage with s_start_1
            temp_v = past_key_values[1][:s_start_len].clone()
        else:
            temp_k = past_key_values[0][s_start_1_len:len(doc_chunk_ids[i])+1].clone()
            temp_v = past_key_values[1][s_start_1_len:len(doc_chunk_ids[i])+1].clone()    

        if j == 0:
            k_tensor = torch.unsqueeze(temp_k, dim=0)
            v_tensor = torch.unsqueeze(temp_v, dim=0)
        else:
            #pdb.set_trace()
            k_tensor = torch.cat((k_tensor,torch.unsqueeze(temp_k, dim=0)), dim=0)
            v_tensor = torch.cat((v_tensor,torch.unsqueeze(temp_v, dim=0)), dim=0)
    k_tensor = torch.unsqueeze(k_tensor, dim=0)
    v_tensor = torch.unsqueeze(v_tensor, dim=0)
    kv_tensor = torch.cat([k_tensor, v_tensor], dim=0)
    cache_driver.collect_kv_and_store(torch.tensor(doc_chunk_ids_store[i]),
                                      kv_tensor.cpu())
    #pdb.set_trace()

cache_driver.dump()
    #print(temp_k.shape[0])
#pdb.set_trace()

    
    
   

