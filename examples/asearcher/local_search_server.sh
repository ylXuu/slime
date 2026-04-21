# python local_search_server.py \
python local_search_server.py \
    --index_path /cpfs01/yilong.xu/datasets/wiki-18-e5-index/e5_Flat.index \
    --corpus_path /cpfs01/yilong.xu/datasets/ASearcher-Local-Knowledge/wiki_corpus.jsonl \
    --pages_path /cpfs01/yilong.xu/datasets/ASearcher-Local-Knowledge/wiki_webpages.jsonl \
    --retriever_name e5 \
    --retriever_model /cpfs01/yilong.xu/models/e5-base-v2 \
    --topk 5 \
    --port 8000
    # --faiss_gpu