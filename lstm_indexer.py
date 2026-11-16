"""
LSTM 知识库索引服务 —— 使用 Siamese LSTM 模型为知识分块构建检索索引
与 original rag_service 的 TF-IDF 方案并存，互不影响

用法：
    from app.services.lstm_indexer import rebuild_course_index_lstm
    rebuild_course_index_lstm(course_id, db)
"""

import uuid
import os
import threading

from app.database import SessionLocal
from app.models.course import CourseMaterial
from app.models.knowledge import KnowledgeChunk
from app.services.parser import parse_document
from app.services.rag_service import chunk_text


UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "static", "materials")


def rebuild_course_index_lstm(course_id: str, db):
    """
    重建课程知识库索引（LSTM 版本）
    与 TF-IDF 版本不同的是，LSTM 不需要预计算向量，
    只需要将知识分块的原始文本存入 KnowledgeChunk 表即可。
    检索时由 LSTMRetriever 实时编码。
    """
    # 1. 删除该课程旧的知识块
    db.query(KnowledgeChunk).filter(KnowledgeChunk.course_id == course_id).delete()
    db.flush()

    # 2. 收集所有已完成解析的材料
    materials = (
        db.query(CourseMaterial)
        .filter(
            CourseMaterial.course_id == course_id,
            CourseMaterial.parse_status != "failed",
        )
        .all()
    )

    # 3. 解析并分块
    all_chunks = []
    for mat in materials:
        try:
            file_path = os.path.join(UPLOAD_DIR, os.path.basename(mat.file_url))
            if not os.path.exists(file_path):
                continue
            with open(file_path, "rb") as f:
                raw_text = parse_document(f.read(), mat.file_type)
            for chunk in chunk_text(raw_text):
                all_chunks.append((mat.id, chunk))
        except Exception:
            continue

    if not all_chunks:
        return

    # 4. 写入知识块（LSTM 不需要 vector_json，留空即可）
    for idx, (mat_id, content) in enumerate(all_chunks):
        db.add(KnowledgeChunk(
            id=str(uuid.uuid4()),
            material_id=mat_id,
            course_id=course_id,
            chunk_index=idx,
            content=content,
            vector_json=None,  # LSTM 不需要预存向量
        ))

    # 5. 更新材料状态
    db.query(CourseMaterial).filter(
        CourseMaterial.course_id == course_id,
    ).update({
        "parse_status": "done",
        "chunk_count": len(all_chunks),
    }, synchronize_session=False)


def process_material_async_lstm(course_id: str, material_id: str):
    """后台异步处理：解析资料 → LSTM 索引重建"""
    db = SessionLocal()
    try:
        material = db.query(CourseMaterial).filter(CourseMaterial.id == material_id).first()
        if not material:
            return

        material.parse_status = "processing"
        db.commit()

        rebuild_course_index_lstm(course_id, db)
        db.commit()

        print(f"[LSTM Indexer] 课程 {course_id[:8]}... 索引完成: {material.file_name}")
    except Exception as e:
        try:
            material = db.query(CourseMaterial).filter(CourseMaterial.id == material_id).first()
            if material:
                material.parse_status = "failed"
                material.error_message = str(e)
                db.commit()
        except Exception:
            pass
        print(f"[LSTM Indexer] 索引失败: {e}")
    finally:
        db.close()


def start_async_lstm_index(course_id: str, material_id: str):
    """启动异步 LSTM 索引线程"""
    t = threading.Thread(
        target=process_material_async_lstm,
        args=(course_id, material_id),
        daemon=True,
    )
    t.start()
