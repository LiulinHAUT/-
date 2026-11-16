"""课程资料上传与管理（LSTM 索引版）—— 与原有 /api/materials 并存"""
import os
import uuid

from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.course import Course, CourseMaterial
from app.utils.course import resolve_course_id
from app.utils.middleware import get_current_user, require_teacher
from app.utils.audit import log_audit
from app.services.lstm_indexer import start_async_lstm_index

router = APIRouter(prefix="/api/materials-lstm", tags=["课程资料-LSTM索引"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "static", "materials")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_TYPES = {
    "text/plain": "txt",
    "text/markdown": "md",
    "text/x-markdown": "md",
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}

MAX_SIZE = 50 * 1024 * 1024  # 50MB


@router.post("/upload/{course_id}")
async def upload_material_lstm(
    course_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_teacher),
):
    """教师为课程上传资料（使用 LSTM 语义检索索引）"""
    course = db.query(Course).filter(Course.slug == course_id).first()
    if not course:
        return {"code": 404, "message": "课程不存在", "data": None}

    content_type = file.content_type or ""
    if content_type not in ALLOWED_TYPES:
        return {"code": 400, "message": f"不支持的文件类型: {content_type}，支持: TXT/MD/PDF/DOCX", "data": None}

    file_type = ALLOWED_TYPES[content_type]
    contents = await file.read()

    if len(contents) > MAX_SIZE:
        return {"code": 400, "message": "文件大小不能超过 50MB", "data": None}

    filename = f"{uuid.uuid4().hex}.{file_type}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(contents)

    material = CourseMaterial(
        id=str(uuid.uuid4()),
        course_id=course.id,
        uploaded_by=current_user["user_id"],
        file_name=file.filename or filename,
        file_type=file_type,
        file_size=len(contents),
        file_url=f"/static/materials/{filename}",
        parse_status="pending",
    )
    db.add(material)
    db.commit()

    # 后台异步 LSTM 索引
    start_async_lstm_index(course.id, material.id)

    log_audit(
        db, current_user["user_id"], current_user["role"],
        "upload_material", target_type="course", target_id=course.id,
        category="course",
        title=f"上传课程资料",
        content=f"教师 {current_user.get('real_name') or current_user.get('username')} 上传资料: {file.filename}",
    )

    return {
        "code": 0, "message": "上传成功! LSTM 索引构建中...",
        "data": {"material_id": material.id, "file_name": file.filename},
    }


@router.get("/{course_id}")
def list_materials_lstm(
    course_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """列出课程的 LSTM 索引资料"""
    cid = resolve_course_id(db, course_id) or course_id
    course = db.query(Course).filter(Course.id == cid).first()
    if not course:
        return {"code": 404, "message": "课程不存在", "data": None}

    materials = (
        db.query(CourseMaterial)
        .filter(CourseMaterial.course_id == cid)
        .order_by(CourseMaterial.created_at.desc())
        .all()
    )

    items = []
    for m in materials:
        items.append({
            "id": m.id,
            "file_name": m.file_name,
            "file_type": m.file_type,
            "file_size": m.file_size,
            "file_url": m.file_url,
            "chunk_count": m.chunk_count,
            "parse_status": m.parse_status,
            "error_message": m.error_message,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        })

    from app.models.knowledge import KnowledgeChunk
    from sqlalchemy import func
    chunk_count = db.query(func.count(KnowledgeChunk.id)).filter(
        KnowledgeChunk.course_id == cid
    ).scalar()

    return {
        "code": 0, "message": "success",
        "data": {
            "items": items,
            "stats": {
                "total_chunks": chunk_count,
                "indexed": chunk_count > 0,
            },
        },
    }
