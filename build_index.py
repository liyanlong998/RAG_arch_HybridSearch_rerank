# -*- coding: utf-8 -*-
"""
建筑规范RAG助手 - 知识库构建（索引入库）

功能：
1. 处理民用建筑设计统一标准规范（文本切片）
2. 处理table文件夹中的HTML表格文件（整个文件作为一个切片）
3. 使用FAISS建立向量索引
"""
import os
import json
import numpy as np
import faiss
import dashscope
from http import HTTPStatus
import time

# 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("错误：请设置 'DASHSCOPE_API_KEY' 环境变量。")

dashscope.api_key = DASHSCOPE_API_KEY

# 如果遇到代理问题，取消下面的注释来禁用代理
# os.environ['NO_PROXY'] = '*'
# os.environ['HTTP_PROXY'] = ''
# os.environ['HTTPS_PROXY'] = ''

# 路径配置
DOCS_DIR = "knowledge_base"
TABLE_DIR = os.path.join(DOCS_DIR, "table")
MULTIMODAL_EMBEDDING_MODEL = "text-embedding-v4"

# 输出文件
INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.json"
CONTENTS_FILE = "contents.json"  # 保存所有内容用于BM25


def parse_text_file(file_path):
    """读取.txt格式文件，提取全部文本"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except UnicodeDecodeError:
        # 如果utf-8解码失败，尝试使用其他编码
        with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
            content = f.read()
        return content


def split_text(text):
    """按两个换行符切分文本"""
    chunks = []
    # 使用两个换行符作为分隔符切分文本
    parts = text.split('\n\n')
    for part in parts:
        if part.strip():
            chunks.append(part.strip())
    return chunks


def get_text_embedding(text, max_retries=3, retry_delay=2):
    """文本embedding，带重试机制"""
    for attempt in range(max_retries):
        try:
            resp = dashscope.TextEmbedding.call(
                model=MULTIMODAL_EMBEDDING_MODEL,
                input=text
            )
            if resp.status_code != HTTPStatus.OK:
                raise Exception(f"文本Embedding失败: {resp.message}")

            # 添加短暂延时，避免请求过于频繁
            time.sleep(0.5)
            return resp.output['embeddings'][0]['embedding']

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    请求失败，{retry_delay}秒后重试... (尝试 {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"    请求失败，已重试{max_retries}次")
                raise e


def build_and_save():
    """构建知识库并保存"""
    print("\n--- 构建建筑规范知识库 ---")
    print(f"规范文档切分: 按两个换行符切分")
    print(f"HTML表格处理: 每个文件一个切片\n")

    metadata_store = []
    all_vectors = []
    doc_id = 0

    # 1. 处理民用建筑设计统一标准规范文本文件（需要切片）
    print("=" * 60)
    print("第一部分: 处理规范文档")
    print("=" * 60)

    for filename in os.listdir(DOCS_DIR):
        if filename.startswith('.') or os.path.isdir(os.path.join(DOCS_DIR, filename)):
            continue

        file_path = os.path.join(DOCS_DIR, filename)
        if filename.lower().endswith(".txt"):
            print(f"\n处理规范文件: {filename}")
            full_text = parse_text_file(file_path)
            chunks = split_text(full_text)
            print(f"  文档长度: {len(full_text)} 字符")
            print(f"  切分为 {len(chunks)} 个chunk")

            for idx, chunk in enumerate(chunks):
                try:
                    metadata = {
                        "id": doc_id,
                        "source": filename,
                        "type": "norm_text",
                        "chunk_index": idx,
                        "content": chunk
                    }

                    vector = get_text_embedding(chunk)
                    all_vectors.append(vector)
                    metadata_store.append(metadata)
                    doc_id += 1

                    if (idx + 1) % 10 == 0:
                        print(f"  已处理 {idx + 1}/{len(chunks)} 个chunk")

                except Exception as e:
                    print(f"  ✗ Chunk {idx} 处理失败: {str(e)}")
                    continue

    # 2. 处理table文件夹中的HTML表格文件（不切片，整个文件作为一个切片）
    print("\n" + "=" * 60)
    print("第二部分: 处理HTML表格文件")
    print("=" * 60)

    if os.path.exists(TABLE_DIR):
        table_files = [f for f in os.listdir(TABLE_DIR) if f.lower().endswith('.html')]
        print(f"\n找到 {len(table_files)} 个HTML表格文件\n")

        for filename in table_files:
            try:
                table_path = os.path.join(TABLE_DIR, filename)

                # 读取HTML表格内容
                table_content = parse_text_file(table_path)

                print(f"处理: {filename}")
                print(f"  文件大小: {len(table_content)} 字符")

                # 构建元数据（整个表格作为一个切片）
                metadata = {
                    "id": doc_id,
                    "source": f"HTML表格: {filename}",
                    "type": "html_table",
                    "table_file": table_path,
                    "table_name": filename,
                    "content": table_content
                }

                # 生成embedding（对HTML内容）
                vector = get_text_embedding(table_content)
                all_vectors.append(vector)
                metadata_store.append(metadata)
                doc_id += 1
                print(f"  ✓ 成功处理\n")

            except Exception as e:
                print(f"  ✗ 处理失败: {str(e)}\n")
                continue
    else:
        print(f"\n警告: {TABLE_DIR} 文件夹不存在")

    # 3. 创建FAISS索引
    print("\n" + "=" * 60)
    print("第三部分: 创建FAISS向量索引")
    print("=" * 60)

    if all_vectors:
        dim = len(all_vectors[0])
        print(f"\n向量维度: {dim}")
        print(f"总条目数: {len(all_vectors)}")

        # 使用L2距离创建FAISS索引
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(all_vectors).astype('float32'))

        # 保存索引
        faiss.write_index(index, INDEX_FILE)
        print(f"\n✓ FAISS索引已保存: {INDEX_FILE}")

        # 保存元数据
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata_store, f, ensure_ascii=False, indent=2)
        print(f"✓ 元数据已保存: {METADATA_FILE}")

        # 保存内容列表用于BM25
        contents_list = [m['content'] for m in metadata_store]
        with open(CONTENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(contents_list, f, ensure_ascii=False, indent=2)
        print(f"✓ 内容列表已保存: {CONTENTS_FILE} (用于BM25检索)")
    else:
        print("\n警告: 没有找到任何数据来构建索引")
        return

    # 4. 统计信息
    print("\n" + "=" * 60)
    print("知识库构建完成统计")
    print("=" * 60)

    norm_count = sum(1 for m in metadata_store if m["type"] == "norm_text")
    table_count = sum(1 for m in metadata_store if m["type"] == "html_table")

    print(f"\n规范文档切片: {norm_count}")
    print(f"HTML表格条目: {table_count}")
    print(f"总计: {len(metadata_store)} 条")

    # 打印部分示例
    print("\n" + "=" * 60)
    print("知识库内容示例 (前5条)")
    print("=" * 60)
    for m in metadata_store[:5]:
        print(f"\n[ID:{m['id']:3d}] 类型: {m['type']}")
        print(f"来源: {m.get('source', '')}")
        if m['type'] == 'html_table' and m.get('table_name'):
            print(f"表格文件: {m['table_name']}")
        content_preview = m['content'][:100] + "..." if len(m['content']) > 100 else m['content']
        print(f"内容: {content_preview}")


if __name__ == "__main__":
    build_and_save()
