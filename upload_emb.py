from huggingface_hub import upload_folder

# 上传 dataset 文件夹到指定子文件夹
upload_folder(
    folder_path="/home/ubuntu/jianwen-us-midwest-1/panlu/zhuofeng-tamu/tevatron/embeddings/browsecomp_plus_fineweb_10BT/qwen3_8b",
    repo_id="ZhuofengLi/fineweb_indexes",
    repo_type="dataset",
    path_in_repo="Sample-10BT-browsecomp-plus/qwen3-embedding-8b",
)
