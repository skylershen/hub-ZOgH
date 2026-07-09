"""
RAG 问答流水线（原生实现）

流程：
  (可选) 查询改写
        ↓
  向量检索（DashScope embedding + FAISS）
        +
  BM25 关键词检索（jieba + rank_bm25）
        ↓
  RRF 融合排名
        ↓
  (可选) CrossEncoder Rerank
        ↓
  相关性阈值过滤（过低则拒绝回答）
        ↓
  LLM 生成（DashScope qwen-plus）+ 引用标注

使用方式：
  python rag_pipeline.py                            # 交互式
  python rag_pipeline.py --query "茅台2023年营收"
  python rag_pipeline.py --query "..." --stock 600519 --year 2023
  python rag_pipeline.py --query "..." --query-rewrite   # 开启查询改写
  python rag_pipeline.py --query "..." --no-bm25 --no-rerank  # 消融测试

依赖：
  pip install faiss-cpu rank_bm25 jieba openai numpy
  export DASHSCOPE_API_KEY="sk-xxx"
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
INDEX_PATH = VECTORSTORE_DIR / "faiss_index.bin"
META_PATH = VECTORSTORE_DIR / "faiss_meta.json"

DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBED_MODEL = "text-embedding-v3"
EMBED_DIM = 1024
LLM_MODEL = "qwen-plus"  # 可换 qwen-turbo（更快）/ qwen-max（更强）

TOP_K_RETRIEVE = 10  # 初始召回数
TOP_K_RERANK = 4  # Rerank 后保留数（给 LLM）
SCORE_THRESHOLD = 0.25  # 最高得分低于此值时触发拒绝回答（归一化余弦相似度）

SYSTEM_PROMPT = """你是一位资深的技术文档专家与高级工程师。你的任务是基于提供的【参考资料】，准确、严谨地解答用户的技术问题。

回答规则：
1. 绝对忠实于原文：你必须且只能基于下方提供的【参考资料】进行回答，严禁依赖你的预训练知识或外部信息。
2. 拒绝幻觉：如果【参考资料】中没有包含回答问题所需的信息，请直接回复：“基于当前提供的文档，未找到相关信息。” 严禁编造任何 API、函数、参数或操作步骤。
3. 保持客观：不要使用“我认为”、“可能”等主观词汇，直接陈述技术事实。
4. 引用溯源：在给出关键结论、配置参数或代码片段时，请在句末使用 [来源：文档编号/章节] 的格式标明出处。
5. 数字要精确，不得四舍五入或模糊表达
6. 回答简洁，重点突出，避免无关废话"""


# ── DashScope 客户端 ──────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("请设置环境变量 DASHSCOPE_API_KEY")
    return OpenAI(api_key=api_key, base_url=DASHSCOPE_URL)


# ── 向量检索 ──────────────────────────────────────────────────────────────────

class VectorStore:
    def __init__(self, client: OpenAI):
        import faiss
        self.client = client
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, encoding="utf-8") as f:
            self.meta_list = json.load(f)
        logger.info(f"FAISS 索引加载完成，共 {self.index.ntotal} 条向量")

    def _embed_query(self, query: str) -> np.ndarray:
        resp = self.client.embeddings.create(
            model=EMBED_MODEL, input=[query], dimensions=EMBED_DIM
        )
        vec = np.array([resp.data[0].embedding], dtype="float32")
        vec = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
        return vec

    def search(
            self,
            query: str,
            top_k: int = TOP_K_RETRIEVE,
            filter_meta: Optional[dict] = None,
    ) -> list[dict]:
        """
        向量检索，可选元数据过滤。
        """
        query_vec = self._embed_query(query)
        # 多取一些再过滤
        scores, indices = self.index.search(query_vec, top_k * 4)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.meta_list):
                continue
            item = dict(self.meta_list[idx])
            item["vec_score"] = float(score)

            if filter_meta:
                if not all(str(item.get(k, "")) == str(v) for k, v in filter_meta.items()):
                    continue

            results.append(item)
            if len(results) >= top_k:
                break
        return results


# ── BM25 关键词检索 ───────────────────────────────────────────────────────────

class BM25Store:
    """
    基于 jieba + rank_bm25 的关键词检索。
    对精确数字、专有名词（如"净利润""研发费用"）效果优于纯向量检索。
    首次初始化会分词整个语料库，约需数秒。
    """

    def __init__(self):
        from rank_bm25 import BM25Okapi
        import jieba

        with open(META_PATH, encoding="utf-8") as f:
            self.meta_list = json.load(f)

        logger.info("构建 BM25 索引（分词中，请稍候）...")
        tokenized = [list(jieba.cut(item["content"])) for item in self.meta_list]
        self.bm25 = BM25Okapi(tokenized)
        self.jieba = jieba
        logger.info("BM25 索引完成")

    def search(self, query: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
        tokens = list(self.jieba.cut(query))
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            if scores[idx] < 1e-9:  # 完全不相关的跳过
                continue
            item = dict(self.meta_list[idx])
            item["bm25_score"] = float(scores[idx])
            results.append(item)
        return results


# ── RRF 融合 ──────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
        vec_results: list[dict],
        bm25_results: list[dict],
        k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion（RRF）。
    公式：score(d) = Σ 1/(k + rank_i(d))，k=60 为经验值。
    将向量召回和 BM25 召回的排名合并，互补各自的盲区。
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, item in enumerate(vec_results, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid] = item

    for rank, item in enumerate(bm25_results, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid] = item

    sorted_cids = sorted(rrf_scores, key=lambda x: -rrf_scores[x])
    results = []
    for cid in sorted_cids:
        item = dict(chunk_map[cid])
        item["rrf_score"] = rrf_scores[cid]
        results.append(item)
    return results


# ── CrossEncoder Rerank（可选）────────────────────────────────────────────────

def rerank(query: str, candidates: list[dict], top_k: int = TOP_K_RERANK) -> list[dict]:
    """
    用 CrossEncoder 对候选集二次精排。
    CrossEncoder 是双向注意力模型，比 bi-encoder 更准确，但只能小批量用。
    模型：BAAI/bge-reranker-base（中文，约 278MB）

    若模型未下载或 sentence-transformers 未安装，自动降级为直接截断。
    """
    try:
        from sentence_transformers import CrossEncoder
        # 如果模型已下载到项目 models/ 目录，优先用本地路径
        model_path = Path(__file__).parent.parent / "models" / "bge-reranker-base"
        model_name = str(model_path) if model_path.exists() else "BAAI/bge-reranker-base"
        reranker = CrossEncoder(model_name)
        pairs = [(query, c["content"]) for c in candidates]
        scores = reranker.predict(pairs)
        for item, score in zip(candidates, scores):
            item["rerank_score"] = float(score)
        candidates.sort(key=lambda x: -x.get("rerank_score", 0))
    except ImportError:
        logger.warning("sentence-transformers 未安装，跳过 Rerank（pip install sentence-transformers）")
    except Exception as e:
        logger.warning(f"Rerank 失败，使用 RRF 原始排序: {e}")

    return candidates[:top_k]


# ── 查询改写 ──────────────────────────────────────────────────────────────────

def rewrite_query(query: str, client: OpenAI) -> str:
    """
    用 LLM 将模糊的用户问题改写为更适合检索的精确表述。
    示例：
      原始：茅台最近怎么样
      改写：贵州茅台2023年营业收入净利润同比增长率经营情况

    使用 qwen-turbo（最快最便宜），不需要高质量推理。
    """
    resp = client.chat.completions.create(
        model="qwen-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一位资深的技术文档专家与高级工程师。将用户的问题改写为更适合从技术文档中检索信息的精确查询语句。"
                    "保留关键实体，扩展相关关键词，不要超过50字。"
                    "直接输出改写后的查询语句，不要解释。"
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0,
    )
    rewritten = resp.choices[0].message.content.strip()
    logger.info(f"查询改写: {query!r} → {rewritten!r}")
    return rewritten


# ── LLM 生成 ──────────────────────────────────────────────────────────────────

def build_context(retrieved: list[dict]) -> tuple[str, list[dict]]:
    """将检索结果组装为 Prompt 上下文，返回上下文字符串和引用列表。"""
    parts = []
    citations = []

    for i, item in enumerate(retrieved, 1):
        page = item.get("page_num", "")
        section = item.get("section", "")

        label = f"[{i}]"
        if section:
            label += f" · {section}"
        if page and page != -1:
            label += f" · 第{page}页"

        # 层级分块时优先用父块内容（信息更完整）
        content = item.get("parent_content") or item.get("content", "")
        parts.append(f"{label}\n{content}")
        citations.append({"index": i, "source": label, "chunk_id": item.get("chunk_id", "")})

    return "\n\n---\n\n".join(parts), citations


def call_llm(query: str, context: str, client: OpenAI) -> str:
    user_msg = (
        f"【参考资料】\n{context}\n\n"
        f"【问题】\n{query}\n\n"
        "请根据参考资料回答，并在引用数据处标注来源编号（如[1]）。"
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content


# ── 完整流水线 ────────────────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(
            self,
            use_bm25: bool = True,
            use_rerank: bool = True,
            use_query_rewrite: bool = False,
    ):
        self.client = get_client()
        self.vec_store = VectorStore(self.client)
        self.use_bm25 = use_bm25
        self.use_rerank = use_rerank
        self.use_qr = use_query_rewrite
        self.bm25_store = BM25Store() if use_bm25 else None

    def query(
            self,
            question: str,
            filter_meta: Optional[dict] = None,
            verbose: bool = False,
    ) -> dict:
        # ① 查询改写（可选）
        retrieval_query = rewrite_query(question, self.client) if self.use_qr else question

        # ② 向量检索
        vec_results = self.vec_store.search(retrieval_query, TOP_K_RETRIEVE, filter_meta)
        if verbose:
            logger.info(
                f"向量召回: {len(vec_results)} 条，最高分={vec_results[0]['vec_score']:.3f}" if vec_results else "向量召回: 0 条")

        # ③ BM25 + RRF 融合
        if self.use_bm25 and self.bm25_store:
            bm25_results = self.bm25_store.search(retrieval_query, TOP_K_RETRIEVE)
            candidates = reciprocal_rank_fusion(vec_results, bm25_results)
            if verbose:
                logger.info(f"BM25 召回: {len(bm25_results)} 条，RRF 后: {len(candidates)} 条")
        else:
            candidates = vec_results

        # ④ Rerank
        if self.use_rerank:
            final = rerank(question, candidates, TOP_K_RERANK)
        else:
            final = candidates[:TOP_K_RERANK]

        if verbose:
            logger.info(f"最终使用 {len(final)} 条上下文")

        # ⑤ 相关性阈值检查
        if not final:
            return {
                "answer": "未找到相关内容，无法回答此问题。",
                "citations": [], "retrieved": [],
            }
        # 阈值始终用 vec_score（余弦相似度，0~1 可解释）；rrf_score 量纲不同不适合做阈值
        top_score = final[0].get("vec_score", final[0].get("rerank_score", 1.0))
        if top_score < SCORE_THRESHOLD and filter_meta is None:
            return {
                "answer": "根据技能知识库未能找到与该问题相关的内容，建议直接查阅原始文档。",
                "citations": [], "retrieved": final,
            }

        # ⑥ LLM 生成
        context, citations = build_context(final)
        answer = call_llm(question, context, self.client)

        return {"answer": answer, "citations": citations, "retrieved": final}


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="技术文档 RAG 问答（原生版）")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--query-rewrite", action="store_true", help="开启查询改写（增加一次 LLM 调用）")
    parser.add_argument("--no-bm25", action="store_true", help="关闭 BM25（消融实验用）")
    parser.add_argument("--no-rerank", action="store_true", help="关闭 Rerank（消融实验用）")
    args = parser.parse_args()

    pipeline = RAGPipeline(
        use_bm25=not args.no_bm25,
        use_rerank=not args.no_rerank,
        use_query_rewrite=args.query_rewrite,
    )

    filter_meta = {}
    if not filter_meta: filter_meta = None

    def print_result(q: str, result: dict):
        print(f"\n{'=' * 60}")
        print(f"问题：{q}")
        print(f"{'=' * 60}")
        print(f"\n{result['answer']}")
        if result["citations"]:
            print("\n── 来源 ──")
            for c in result["citations"]:
                print(f"  {c['source']}")

    if args.query:
        result = pipeline.query(args.query, filter_meta=filter_meta, verbose=True)
        print_result(args.query, result)
    else:
        print("技术文档 RAG 问答系统（原生版）")
        print(f"模型：{LLM_MODEL}  |  向量库：{INDEX_PATH}")
        print("输入 'exit' 退出，'mode' 查看当前配置\n")
        while True:
            try:
                q = input("问题：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() == "exit":
                break
            if q.lower() == "mode":
                print(f"BM25={'on' if pipeline.use_bm25 else 'off'}  "
                      f"Rerank={'on' if pipeline.use_rerank else 'off'}  "
                      f"QueryRewrite={'on' if pipeline.use_qr else 'off'}")
                continue
            result = pipeline.query(q, filter_meta=filter_meta, verbose=True)
            print_result(q, result)


if __name__ == "__main__":
    main()
