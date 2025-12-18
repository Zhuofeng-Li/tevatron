# CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.vllm_encode \
#   --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
#   --dataset_name Tevatron/browsecomp-plus-corpus \
#   --encode_output_path embeddings/corpus.pkl \
#   --passage_max_len 4096 \
#   --per_device_eval_batch_size 1024 \

GPUS=(4 5 6 7)
NUM_SHARDS=${#GPUS[@]}

for i in "${!GPUS[@]}"; do
  gpuid=${GPUS[$i]}
  shard_index=$i   

  CUDA_VISIBLE_DEVICES=$gpuid \
  python -m tevatron.retriever.driver.vllm_encode \
    --model_name_or_path Qwen/Qwen3-Embedding-8B \
    --dataset_name ZhuofengLi/fineweb_corpus \
    --dataset_config Sample-10BT \
    --encode_output_path fineweb_10BT_emb_tmp/qwen3_8B/corpus.${shard_index}.pkl \
    --num_proc 32 \
    --passage_max_len 4096 \
    --per_device_eval_batch_size 2048 \
    --dataset_number_of_shards $NUM_SHARDS \
    --dataset_shard_index $shard_index &
done

# Wait for all background processes to complete
wait

