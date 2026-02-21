# -*- coding: utf-8 -*-
"""
建筑规范RAG助手 - 查询处理

功能：加载索引，处理用户query，打印相似度排名，支持表格关键词检测
"""
import os
import json
import numpy as np
import faiss
import dashscope
from http import HTTPStatus
from openai import OpenAI
import time
import jieba
from rank_bm25 import BM25Okapi
from typing import List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("错误：请设置 'DASHSCOPE_API_KEY' 环境变量。")

dashscope.api_key = DASHSCOPE_API_KEY

# 如果遇到代理问题，取消下面的注释来禁用代理
# os.environ['NO_PROXY'] = '*'
# os.environ['HTTP_PROXY'] = ''
# os.environ['HTTPS_PROXY'] = ''

client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

MULTIMODAL_EMBEDDING_MODEL = "text-embedding-v4"
INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.json"
CONTENTS_FILE = "contents.json"  # 存储所有内容用于BM25

# 关键词配置
TABLE_KEYWORDS = ["表格", "数据", "表", "列表"]
MEDIA_DISTANCE_THRESHOLD = 3.0  # 表格匹配的距离阈值

# 混合检索参数
ALPHA = 0.5  # 向量检索权重 (0-1), BM25权重为 1-alpha

# Rerank配置
RERANK_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "BAAI", "bge-reranker-base")
HYBRID_RECALL_K = 10   # 混合检索召回条数
RERANK_TOP_K = 3       # rerank后最终保留条数


class Reranker:
    """基于本地模型的Rerank精排器"""

    def __init__(self, model_dir: str = RERANK_MODEL_DIR):
        print(f"正在加载Rerank模型: {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Rerank模型已加载，使用设备: {self.device}")

    def rerank(self, query: str, candidates: List[dict], top_k: int = RERANK_TOP_K) -> List[dict]:
        """
        对候选结果列表进行精排

        参数:
            query: 查询文本
            candidates: hybrid_search 返回的结果列表（每项含 metadata、hybrid_score 等字段）
            top_k: 返回前k个结果

        返回:
            按rerank分数降序排列的结果列表，每项新增 rerank_score 字段
        """
        if not candidates:
            return []

        pairs = [[query, c["metadata"]["content"]] for c in candidates]

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            scores = self.model(**inputs).logits.squeeze(-1).cpu().tolist()

        if isinstance(scores, float):
            scores = [scores]

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = score

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]


def tokenize_chinese(text: str) -> List[str]:
    """中文分词"""
    return list(jieba.cut(text))


def load_index():
    """加载索引、元数据和内容列表（用于BM25）"""
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # 加载内容列表用于BM25
    contents = []
    if os.path.exists(CONTENTS_FILE):
        with open(CONTENTS_FILE, 'r', encoding='utf-8') as f:
            contents = json.load(f)
        print(f"已加载索引: {index.ntotal} 条记录（包含BM25支持）")
    else:
        # 如果contents.json不存在，从metadata提取
        contents = [m['content'] for m in metadata]
        print(f"已加载索引: {index.ntotal} 条记录（从metadata提取内容）")

    # 构建BM25索引
    tokenized_contents = [tokenize_chinese(content) for content in contents]
    bm25 = BM25Okapi(tokenized_contents)

    return index, metadata, contents, bm25


def get_text_embedding(text, max_retries=3, retry_delay=2):
    """文本embedding，带重试机制"""
    for attempt in range(max_retries):
        try:
            resp = dashscope.TextEmbedding.call(
                model=MULTIMODAL_EMBEDDING_MODEL,
                input=text
            )
            if resp.status_code != HTTPStatus.OK:
                raise Exception(f"Embedding失败: {resp.message}")

            # 添加短暂延时，避免请求过于频繁
            time.sleep(0.3)
            return resp.output['embeddings'][0]['embedding']

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"请求失败，{retry_delay}秒后重试... (尝试 {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"请求失败，已重试{max_retries}次")
                raise e


def distance_to_similarity(distance):
    """L2距离转相似度 (0-1之间，越大越相似)"""
    return 1 / (1 + distance)


def detect_media_intent(query):
    """检测query中是否包含表格意图"""
    query_lower = query.lower()
    want_table = any(kw in query_lower for kw in TABLE_KEYWORDS)
    return want_table


def hybrid_search(query, index, metadata, contents, bm25, alpha=ALPHA):
    """
    混合检索：BM25 + 向量检索

    参数:
        query: 查询文本
        index: FAISS索引
        metadata: 元数据列表
        contents: 内容列表
        bm25: BM25索引
        alpha: 向量检索权重 (0-1), BM25权重为 1-alpha

    返回:
        排序后的结果列表，每个元素包含索引、距离、相似度和元数据
    """
    # 1. BM25检索
    tokenized_query = tokenize_chinese(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    # 归一化BM25分数
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    bm25_scores_normalized = [s / max_bm25 for s in bm25_scores]

    # 2. 向量检索
    query_vec = np.array([get_text_embedding(query)]).astype('float32')
    distances, indices = index.search(query_vec, index.ntotal)

    # 构建向量分数字典（距离转相似度）
    vector_scores = {}
    max_distance = max(distances[0]) if len(distances[0]) > 0 else 1
    for idx, dist in zip(indices[0], distances[0]):
        if idx != -1:
            # 距离转分数: 1 - (distance / max_distance)
            vector_scores[idx] = 1 - (dist / max_distance) if max_distance > 0 else 0

    # 3. 融合分数
    hybrid_scores = []
    for idx in range(len(contents)):
        bm25_score = bm25_scores_normalized[idx]
        vector_score = vector_scores.get(idx, 0)
        combined_score = alpha * vector_score + (1 - alpha) * bm25_score

        # 获取原始距离
        original_distance = distances[0][list(indices[0]).index(idx)] if idx in indices[0] else float('inf')

        hybrid_scores.append({
            "idx": idx,
            "distance": original_distance,
            "similarity": distance_to_similarity(original_distance),
            "hybrid_score": combined_score,
            "bm25_score": bm25_score,
            "vector_score": vector_score,
            "metadata": metadata[idx]
        })

    # 按混合分数排序
    hybrid_scores.sort(key=lambda x: x["hybrid_score"], reverse=True)

    return hybrid_scores


def search_with_details(query, index, metadata, contents, bm25):
    """使用混合检索并打印相似度详情，返回top-10候选"""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    # 使用混合检索，获取全部结果
    results = hybrid_search(query, index, metadata, contents, bm25, alpha=ALPHA)

    # 打印top-10
    top10 = results[:HYBRID_RECALL_K]
    print(f"\n混合检索Top-{HYBRID_RECALL_K}排名 (BM25 + 向量, alpha={ALPHA}):")
    print("-" * 100)
    print(f"{'排名':4s} {'ID':4s} {'类型':6s} {'混合分':8s} {'BM25':8s} {'向量':8s} {'距离':8s} 内容")
    print("-" * 100)

    for rank, r in enumerate(top10):
        m = r['metadata']
        content_preview = m['content'][:45].replace('\n', ' ')
        type_tag = m['type']

        marker = ""
        if type_tag == "html_table":
            marker = " <-- HTML表格"

        print(f"{rank+1:4d} {r['idx']:4d} [{type_tag:12s}] {r['hybrid_score']:6.4f}  {r['bm25_score']:6.4f}  {r['vector_score']:6.4f}  {r['distance']:8.4f}  {content_preview}...{marker}")

    return results


def rag_ask(query, index, metadata, contents, bm25, reranker, k=RERANK_TOP_K):
    """RAG问答，混合检索召回10条后经rerank精排，取top-k规范文本，支持表格检测"""
    results = search_with_details(query, index, metadata, contents, bm25)

    # 检测表格意图
    want_table = detect_media_intent(query)
    print(f"\n意图检测: 需要表格={want_table}")

    # ---- 第一阶段：从top-10中取norm_text候选交给reranker ----
    norm_candidates = [r for r in results[:HYBRID_RECALL_K] if r["metadata"]["type"] == "norm_text"]
    print(f"\nRerank阶段：对{len(norm_candidates)}条norm_text候选进行精排...")
    norm_results = reranker.rerank(query, norm_candidates, top_k=k)

    print(f"\nRerank后Top-{k}规范文本:")
    print("-" * 80)
    for rank, r in enumerate(norm_results):
        print(f"  {rank+1}. rerank={r['rerank_score']:.4f}  hybrid={r['hybrid_score']:.4f}  "
              f"{r['metadata']['content'][:50].replace(chr(10), ' ')}...")
    print("-" * 80)

    # ---- 第二阶段：表格匹配仍从全部结果中按距离筛选 ----
    matched_table = None
    table_result = None
    if want_table:
        table_results = [r for r in results if r["metadata"]["type"] == "html_table" and r["distance"] < MEDIA_DISTANCE_THRESHOLD]
        if table_results:
            table_results.sort(key=lambda x: x["distance"])
            matched_table = table_results[0]
            table_result = matched_table
            table_path = matched_table['metadata'].get('table_file', '未找到表格')
            table_name = matched_table['metadata'].get('table_name', '未知')
            print(f"  -> 匹配到表格: {table_name} (距离: {matched_table['distance']:.4f}, 混合分数: {matched_table['hybrid_score']:.4f})")
            print(f"     路径: {table_path}")

    # 构建context
    context_str = ""
    knowledge_count = 0

    # 添加rerank后的规范文本
    for r in norm_results:
        knowledge_count += 1
        m = r["metadata"]
        context_str += f"背景知识 {knowledge_count} [规范文档] (来源: {m['source']}, rerank分数: {r['rerank_score']:.4f}):\n{m['content']}\n\n"

    # 添加表格内容（如果有）
    if table_result:
        knowledge_count += 1
        m = table_result["metadata"]
        context_str += f"背景知识 {knowledge_count} [表格数据] (来源: {m['source']}, 混合分数: {table_result['hybrid_score']:.4f}):\n{m['content']}\n\n"

    prompt = f"""你是一个建筑设计知识问答助手。。

[背景知识]
{context_str}
[用户问题]
{query}
"""

    # 调用LLM
    print("\n调用LLM生成答案...")
    completion = client.chat.completions.create(
        model="deepseek-v3.2",
        messages=[
            {"role": "system", "content": "你是一个建筑设计知识问答助手，回答问题要简洁，如果没有在知识库中检索到相应的内容，不要自己发挥。如果检索到的内容有表格，不需要在回答中展示表格"},
            {"role": "user", "content": prompt}
        ]
    )
    answer = completion.choices[0].message.content

    # 附加匹配到的表格
    if matched_table:
        table_path = matched_table['metadata'].get('table_file')
        table_name = matched_table['metadata'].get('table_name', '未知表格')
        if table_path:
            answer += f"\n\n[相关表格]: {table_name}\n[表格路径]: {table_path}"
        else:
            answer += f"\n\n[相关表格]: {table_name} (表格文件未找到)"

    print(f"\n最终答案:\n{answer}")
    return answer


if __name__ == "__main__":
    index, metadata, contents, bm25 = load_index()
    reranker = Reranker()

    '''
    print("\n" + "="*60)
    print("测试: 混合检索 + Rerank")
    print("="*60)
    rag_ask("楼梯踏步的尺寸要求是什么？请提供相关表格", index, metadata, contents, bm25, reranker)
    '''

    print("\n" + "="*60)
    print("测试: 混合检索 + Rerank")
    print("="*60)
    rag_ask("金属夹芯板的屋面排水坡度是多少？请提供相关表格", index, metadata, contents, bm25, reranker)


