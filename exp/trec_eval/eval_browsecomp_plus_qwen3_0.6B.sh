# CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.vllm_encode \
#   --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
#   --dataset_path data/browsecomp_plus_decrypted.jsonl \
#   --encode_output_path embeddings/browsecomp_plus_query/qwen3_0.6b/query.pkl \
#   --query_max_len 512 \
#   --encode_is_query \
#   --num_proc 32 \
#   --encode_is_query \
#   --query_prefix "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:" \
#   --dataloader_num_workers 32 \
#   --per_device_eval_batch_size 1024 


huggingface-cli download Tevatron/browsecomp-plus-indexes --repo-type=dataset --include="qwen3-embedding-0.6b/*" --local-dir ./embeddings/browsecomp_plus

mkdir -p runs
python -m tevatron.retriever.driver.search --query_reps "embeddings/browsecomp_plus_query/qwen3_0.6b/query.pkl" --passage_reps "embeddings/browsecomp_plus/qwen3-embedding-0.6b/*.pkl" --depth 1000 --batch_size 128 --save_text --save_ranking_to runs/qwen3-0.6b_top1000.txt

python -m tevatron.utils.format.convert_result_to_trec --input runs/qwen3-0.6b_top1000.txt --output runs/qwen3-0.6b_top1000.trec

# Retrieval Results (Evidence)
python -m pyserini.eval.trec_eval -c -m recall.5,100,1000 -m ndcg_cut.10 examples/BrowseComp-Plus/topics-qrels/qrel_evidence.txt runs/qwen3-0.6b_top1000.trec

# Retrieval Results (Gold)
python -m pyserini.eval.trec_eval  -c -m recall.5,100,1000  -m ndcg_cut.10   examples/BrowseComp-Plus/topics-qrels/qrel_golds.txt  runs/qwen3-0.6b_top1000.trec