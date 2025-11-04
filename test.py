from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from dataclasses import dataclass
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool, ToolRuntime
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import bs4
import getpass
import os
from dotenv import load_dotenv

import numpy as np
from collections import deque
from typing import Any


load_dotenv(override=True)
os.environ["LANGCHAIN_TRACING_V2"] = "true"

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")






# =========================
# ✅ 定义 LangGraph 的记忆系统
# =========================
# InMemorySaver 用于在运行时保存 agent 的上下文（比如对话历史）
# 如果你需要在不同会话间持久化，可以换成 RedisSaver 或 SQLiteSaver

checkpointer = InMemorySaver()

# =========================
# ✅ 初始化大语言模型
# =========================
# 使用 gpt-4o-mini 模型（比 gpt-4o 更快更便宜）

model=init_chat_model(
         "gpt-4o-mini",
          temperature=0.5,
          timeout=10,
          max_tokens=1000
    )

# =========================
# ✅ 定义输出数据结构（Response Schema）
# =========================
# 这部分定义了 Agent 输出的结构化格式，用于类型化返回结果

@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None





# =========================
# ✅ 定义rag向量存储
# =========================


vector_store = InMemoryVectorStore(embeddings)


bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

assert len(docs) == 1
print(f"爬出来了！Total characters: {len(docs[0].page_content)}")




class SemanticCache:
    """
    语义缓存（简单 LRU + embedding 相似度判断）。
    存储结构：
      - queries: 原始 query 文本（list）
      - embeddings: numpy array list
      - responses: 序列化字符串（或你希望缓存的任意对象）
      - docs: 对应的检索到的 Document 列表（可选）
      - order: deque, 用于实现 LRU 淘汰（保存索引）
    使用方法：
      cache = SemanticCache(embeddings_model=embeddings, max_size=128, threshold=0.88)
      hit = cache.get(query)  # 如果命中返回 dict 否则 None
      cache.add(query, serialized, retrieved_docs)
    """
    def __init__(self, embeddings_model, max_size: int = 128, threshold: float = 0.88):
        self.emb_model = embeddings_model
        self.max_size = max_size
        self.threshold = float(threshold)  # 相似度阈值（cosine）
        self.queries: list[str] = []
        self.embeddings: list[np.ndarray] = []
        self.responses: list[Any] = []    # 存放序列化结果（string 或其他）
        self.docs: list[Any] = []         # 存放对应的 retrieved docs（Document 列表）
        self.order = deque()              # 保存索引，实现 LRU：右侧为最近使用

    def _embed(self, text: str) -> np.ndarray:
        # OpenAIEmbeddings 通常有 embed_documents 或 embed_query
        # 我们优先尝试 embed_documents（返回 list），否则尝试 embed_query
        try:
            emb = self.emb_model.embed_documents([text])[0]
        except Exception:
            emb = self.emb_model.embed_query(text)  # 某些实现存在该方法
        return np.array(emb, dtype=float)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def find_best(self, emb: np.ndarray):
        """返回 (best_idx, best_score) 或 (None, 0.0)"""
        if len(self.embeddings) == 0:
            return None, 0.0
        best_idx = None
        best_score = -1.0
        # 线性扫描：缓存条目通常不多（max_size <= 1024），足够快
        for i, e in enumerate(self.embeddings):
            score = self._cosine(emb, e)
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx, best_score

    def get(self, query: str):
        """尝试命中缓存。命中返回 dict: {'query', 'response', 'docs', 'score'}"""
        emb = self._embed(query)
        best_idx, best_score = self.find_best(emb)
        if best_idx is not None and best_score >= self.threshold:
            # 更新 LRU：把 best_idx 移到右端（最近使用）
            try:
                self.order.remove(best_idx)
            except ValueError:
                pass
            self.order.append(best_idx)
            return {
                "query": self.queries[best_idx],
                "response": self.responses[best_idx],
                "docs": self.docs[best_idx],
                "score": best_score,
            }
        return None

    def add(self, query: str, response: Any, docs: Any):
        """把新条目加入缓存，若超过 max_size 则淘汰最旧的条目"""
        emb = self._embed(query)
        # 添加
        idx = len(self.queries)
        self.queries.append(query)
        self.embeddings.append(emb)
        self.responses.append(response)
        self.docs.append(docs)
        self.order.append(idx)
        # 淘汰超出上限的最旧项
        while len(self.order) > self.max_size:
            old_idx = self.order.popleft()
            # 标记删除：将位置留空以保持索引稳定（简单实现）
            # 这里我们将对应条目置为 None; 下次 find_best 时会跳过 None
            self.queries[old_idx] = None
            self.embeddings[old_idx] = np.zeros_like(emb) * 0.0
            self.responses[old_idx] = None
            self.docs[old_idx] = None

    def stats(self):
        total = sum(1 for q in self.queries if q is not None)
        return {"capacity": self.max_size, "entries": total, "threshold": self.threshold}
    
    
# 创建语义缓存实例
semantic_cache = SemanticCache(embeddings_model=embeddings, max_size=128, threshold=0.3) # 创建语义缓存实例



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"分割博客文章为 {len(all_splits)} 个子文档。")

document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])







# =========================
# ✅ 定义可供 Agent 使用的工具函数（Tools）
# =========================
# @tool(response_format="content_and_artifact")
# def retrieve_context(query: str):
#     """Retrieve information to help answer a query."""
#     retrieved_docs = vector_store.similarity_search(query, k=2)
#     serialized = "\n\n".join(
#         (f"Source: {doc.metadata}\nContent: {doc.page_content}")
#         for doc in retrieved_docs
#     )
#     return serialized, retrieved_docs     #旧版本的

# =========================

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """
    语义缓存优先策略：
      1) 先在 semantic_cache 尝试命中（emb + cosine >= threshold）
      2) 若命中，返回缓存中的序列化内容和 docs（节省检索开销）
      3) 若未命中，走原向量库检索 vector_store.similarity_search(...)
           并把序列化结果写入缓存
    """
    # 1) 尝试语义缓存命中
    hit = semantic_cache.get(query)
    if hit is not None:
        # 返回缓存命中结果，注意与原返回格式保持一致 (serialized, retrieved_docs)
        print(f"[Cache] HIT (score={hit['score']:.3f}) for query: {query[:50]}...")
        return hit["response"], hit["docs"]

    # 2) 缓存未命中 -> 真正检索
    print("[Cache] MISS -> doing vector store similarity_search...")
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {d.metadata}\nContent: {d.page_content}") for d in retrieved_docs
    )
    # 3) 把结果加入缓存
    semantic_cache.add(query, serialized, retrieved_docs)
    return serialized, retrieved_docs


@tool
def search_web(query: str) -> str:
    """Search the web for information about the query."""
    return f"模拟搜索结果: '{query}'"



# =========================
# ✅ 定义自定义上下文（Context Schema）
# =========================
# Agent 执行时可以访问这个上下文，比如用户 ID、权限、配置等

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

# =========================
# ✅ 创建智能代理（Agent）
# =========================
# create_agent 用于组装一个可执行的智能体，
# 它能自动根据系统提示、工具和上下文来规划任务。
prompt = """
You are an AI Agent with access to retrieval tools.
You are an AI expert assistant specialized in answering questions in AI / Machine Learning / LLM / RAG / LangChain domain.
Your goal is to answer the user's question as concise, accurate and helpful as possible.

### RULES
- You may call the `retrieve_context` tool **at most once** per user query.
- If you think you already have enough knowledge to answer, DO NOT call the tool.
- If the retrieval tool returns context, use it to synthesize the final short answer.
- Never enter infinite loops, repeated tool calls, or repeated self-queries.
- Always return the final answer in the ResponseFormat schema.

### Output Format Reminder
- punny_response: must be playful / witty / pun style
- weather_conditions: only set if the user explicitly asked for weather or relevant

### Goal
Answer user queries efficiently using context from the blog if needed.
When uncertain → retrieve once → integrate → answer. 
When confident → answer directly.
"""


agent = create_agent(
    model=model,                        # 语言模型
    system_prompt=prompt,  # 系统角色提示
    tools=[search_web,retrieve_context],           # 可用工具
    context_schema=Context,             # 上下文类型定义
    response_format=ResponseFormat,     # 输出格式定义
    checkpointer=checkpointer           # 内存检查点（存储对话状态）
)

# =========================
# ✅ 定义 Agent 调用配置
# =========================
# “thread_id” 用于标识一个连续的会话线程（支持多轮记忆）

config = {"configurable": {"thread_id": "1"}}



print("启动多轮交互（输入 'exit' 退出）")

while True:
    user_input = input("\n你: ")
    if user_input.strip().lower() in ("exit", "quit"):
        print("已退出。")
        break

    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
        context=Context(user_id="1")
    )

    # 结构化返回：按 ResponseFormat 使用
    structured = response.get("structured_response")
    # 保险打印：若是 dataclass / object，尽量访问属性
    if structured:
        try:
            print("\n助手:", structured.punny_response)
            if getattr(structured, "weather_conditions", None):
                print("天气信息:", structured.weather_conditions)
        except Exception:
            # 兜底打印整个结构
            print("\n助手(完整结构):", structured)
    else:
        print("\n助手（原始响应）:", response)








