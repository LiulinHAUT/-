"""RAG 语义检索 API —— 基于训练好的 Siamese LSTM 模型
路由前缀 /api/rag-lstm，与原 /api/rag（TF-IDF）并存，互不影响

接入方式：在 app/main.py 中添加一行即可:
    from app.routers.rag_lstm import router as rag_lstm_router
    app.include_router(rag_lstm_router)
"""
import json

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.utils.course import resolve_course_id
from app.models.knowledge import KnowledgeChunk, CourseCorpus
from app.utils.middleware import get_current_user
from app.services.lstm_rag_service import get_lstm_retriever

router = APIRouter(prefix="/api/rag-lstm", tags=["RAG检索-LSTM语义"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


def _load_chunks(db: Session, course_id: str) -> list:
    """从数据库加载课程的所有知识分块"""
    cid = resolve_course_id(db, course_id) or course_id
    chunks = (
        db.query(KnowledgeChunk)
        .filter(KnowledgeChunk.course_id == cid)
        .order_by(KnowledgeChunk.chunk_index)
        .all()
    )
    return chunks


@router.post("/search/{course_id}")
def search_course_knowledge_lstm(
    course_id: str,
    req: SearchRequest,
    mode: str = Query("lstm", description="检索模式: lstm | tfidf | compare（同时返回两种结果）"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    在课程知识库中检索与查询最相关的知识分块

    支持三种模式:
      - lstm:  使用 Siamese LSTM 语义向量检索（默认，效果好）
      - tfidf: 使用原 TF-IDF 稀疏向量检索（兼容）
      - compare: 同时返回两种结果，用于对比展示
    """
    chunks = _load_chunks(db, course_id)
    if not chunks:
        return {"code": 0, "message": "success", "data": {"results": [], "total_chunks": 0, "mode": mode}}

    contents = [c.content for c in chunks]

    if mode == "compare":
        # 同时返回两种检索结果
        result_lstm = _search_lstm(req.query, contents, req.top_k)
        result_tfidf = _search_tfidf(db, course_id, req.query, chunks, contents, req.top_k)
        return {
            "code": 0, "message": "success",
            "data": {
                "mode": "compare",
                "lstm_results": result_lstm,
                "tfidf_results": result_tfidf,
                "total_chunks": len(chunks),
            },
        }

    elif mode == "tfidf":
        results = _search_tfidf(db, course_id, req.query, chunks, contents, req.top_k)
        return {"code": 0, "message": "success", "data": {"results": results, "total_chunks": len(chunks), "mode": "tfidf"}}

    else:  # mode == "lstm"
        results = _search_lstm(req.query, contents, req.top_k)
        return {"code": 0, "message": "success", "data": {"results": results, "total_chunks": len(chunks), "mode": "lstm"}}


def _search_lstm(query: str, contents: list, top_k: int) -> list:
    """使用 Siamese LSTM 进行语义检索"""
    try:
        retriever = get_lstm_retriever()
        retriever.build_index(contents)  # 每门课首次调用时构建索引
        return retriever.search(query, top_k=top_k)
    except Exception as e:
        return [{"error": f"LSTM检索失败: {e}"}]


def _search_tfidf(db: Session, course_id: str, query: str,
                   chunks: list, contents: list, top_k: int) -> list:
    """使用原有 TF-IDF 方法检索（作为对比基线）"""
    import numpy as np
    from app.services.rag_service import (
        encode_query, search_top_k, sparse_json_rows_to_matrix,
    )

    cid = resolve_course_id(db, course_id) or course_id
    corpus = db.query(CourseCorpus).filter(CourseCorpus.course_id == cid).first()
    if not corpus:
        return [{"error": "TF-IDF索引不存在"}]

    vectors_json = [c.vector_json for c in chunks if c.vector_json]
    if not vectors_json:
        return [{"error": "无TF-IDF向量数据"}]

    try:
        chunk_vectors = sparse_json_rows_to_matrix(vectors_json)
        query_vec = encode_query(query, corpus.vocab_json)
        return search_top_k(query_vec, chunk_vectors, contents, top_k=top_k)
    except Exception as e:
        return [{"error": f"TF-IDF检索失败: {e}"}]


@router.get("/status/{course_id}")
def get_rag_status_lstm(
    course_id: str,
    db: Session = Depends(get_db),
):
    """查询课程知识库状态（含LSTM模型状态）"""
    chunks = _load_chunks(db, course_id)
    cid = resolve_course_id(db, course_id) or course_id
    corpus = db.query(CourseCorpus).filter(CourseCorpus.course_id == cid).first()

    retriever = get_lstm_retriever()
    retriever._ensure_loaded()

    return {
        "code": 0, "message": "success",
        "data": {
            "chunk_count": len(chunks),
            "tfidf_indexed": corpus is not None,
            "vocab_size": len(json.loads(corpus.vocab_json).get("vocab", {})) if corpus else 0,
            "lstm_loaded": retriever.is_loaded,
            "lstm_embedding_dim": retriever.embedding_dim,
        },
    }
