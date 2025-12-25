from huggingface_hub import upload_folder

# 上传 dataset 文件夹到指定子文件夹
upload_folder(
    folder_path="/home/ubuntu/jianwen-us-midwest-1/panlu/zhuofeng-128/tevatron/embeddings/miroverse-10k/qwen3_8b",
    repo_id="ZhuofengLi/fineweb_indexes",
    repo_type="dataset",
    path_in_repo="miroverse-10k/qwen3-embedding-8b",
)
