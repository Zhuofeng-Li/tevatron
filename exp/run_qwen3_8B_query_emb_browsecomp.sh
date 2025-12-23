# Note: only use one GPU to encode the queries
CUDA_VISIBLE_DEVICES=2  python -m tevatron.retriever.driver.vllm_encode \
  --model_name_or_path Qwen/Qwen3-Embedding-8B \
  --dataset_path data/browsecomp_plus_decrypted.jsonl \
  --encode_output_path embeddings/browsecomp_plus_query/qwen3_8b/query.pkl \
  --num_proc 32 \
  --query_max_len 512 \
  --encode_is_query \
  --query_prefix "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:" \
  --per_device_eval_batch_size 1024 \