GPUS=(0 1 2 3 4 5 6 7)
NUM_SHARDS=${#GPUS[@]}

for i in "${!GPUS[@]}"; do
  gpuid=${GPUS[$i]}
  shard_index=$i

  CUDA_VISIBLE_DEVICES=$gpuid python -m tevatron.retriever.driver.vllm_encode \
    --model_name_or_path Qwen/Qwen3-Embedding-8B \
    --dataset_name ZhuofengLi/fineweb_corpus \
    --dataset_config miroverse-10k \
    --encode_output_path embeddings/miroverse-10k/qwen3_8b/corpus.${shard_index}.pkl \
    --num_proc 128 \
    --dataloader_num_workers 128 \
    --passage_max_len 4096 \
    --per_device_eval_batch_size 1024 \
    --dataset_number_of_shards $NUM_SHARDS \
    --dataset_shard_index $shard_index &
done
wait

