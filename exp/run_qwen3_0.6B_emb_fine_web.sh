# CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.vllm_encode \
#   --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
#   --dataset_name Tevatron/browsecomp-plus-corpus \
#   --encode_output_path embeddings/corpus.pkl \
#   --passage_max_len 4096 \
#   --per_device_eval_batch_size 1024 \

for s in 2 3
do
gpuid=$s
CUDA_VISIBLE_DEVICES=$gpuid python -m tevatron.retriever.driver.vllm_encode \
  --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
  --dataset_name fineweb/Sample-10BT \
  --encode_output_path fineweb_10BT_emb_tmp/corpus.${s}.pkl \
  --num_proc 32 \
  --passage_max_len 4096 \
  --per_device_eval_batch_size 1024 \
  --dataset_number_of_shards 8 \
  --dataset_shard_index $s &
done

# Wait for all background processes to complete
wait

