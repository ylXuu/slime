#!/bin/bash

# 配置参数
URL="http://127.0.0.1:8000/retrieve"
CONCURRENCY=20  # 在这里调节并发数
TOTAL_REQUESTS=40 # 总请求数

echo "开始压测: 并发数=$CONCURRENCY, 总请求=$TOTAL_REQUESTS"

for ((i=1; i<=$TOTAL_REQUESTS; i++)); do
  curl -s -w "请求 $i - 状态码: %{http_code} - 耗时: %{time_total}s\n" \
    -o /dev/null \
    -X POST "$URL" \
    -H "Content-Type: application/json" \
    -d '{"queries": ["Python programming language"], "topk": 3}' &

  # 达到并发数上限时等待
  if (( i % CONCURRENCY == 0 )); then
    wait
  fi
done

wait
echo "压测完成。"