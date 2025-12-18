## Installation
```bash 
uv venv --python 3.12
source .venv/bin/activate 
uv pip install transformers datasets peft deepspeed accelerate faiss-cpu vllm 
uv pip install -e .
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.15/flash_attn-2.8.3+cu126torch2.9-cp312-cp312-linux_x86_64.whl
```

## Embedding Gen 
```bash
# fineweb 10BT corpus 
hf download ZhuofengLi/fineweb_corpus --repo-type dataset --local-dir "./fineweb" --include "Sample-10BT/*"

# gen finweb 10BT emb using qwen3-0.6B  
bash exp/run_qwen3_0.6B_emb_fine_web.sh
```

## RAG Eval 
### Setup dataset and emb (query and corpus)
```bash 
for s in 2 3 4 5 6
do
gpuid=$s
CUDA_VISIBLE_DEVICES=$gpuid python -m tevatron.retriever.driver.vllm_encode \
  --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
  --dataset_path data/browsecomp_plus_decrypted.jsonl \
  --encode_output_path embeddings/query.{s}.pkl \
  --query_max_len 512 \
  --encode_is_query \
  --num_proc 32 \
  --per_device_eval_batch_size 1024 \
  --dataset_number_of_shards 5 \
  --dataset_shard_index $s &
done
```
