"""Main FastAPI application for LightRAG-DuckDB."""

import os
from pathlib import Path

from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import config
import database as db
import indexer
import retriever
from embeddings import EmbeddingError
from llm import LLMError

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="LightRAG-DuckDB", description="Graph-based RAG on DuckDB with MiniMax")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _nav_context(request: Request, page: str) -> dict:
    cfg = config.load_config()
    return {
        "request": request,
        "page": page,
        "has_api_key": bool(cfg.get("minimax_api_key")),
        "stats": db.get_stats(),
    }


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.get("/")
async def index_page(request: Request):
    return templates.TemplateResponse(request, "index.html", _nav_context(request, "query"))


@app.get("/upload")
async def upload_page(request: Request):
    ctx = _nav_context(request, "upload")
    ctx["documents"] = db.list_documents()
    return templates.TemplateResponse(request, "upload.html", ctx)


@app.get("/graph")
async def graph_page(request: Request):
    return templates.TemplateResponse(request, "graph.html", _nav_context(request, "graph"))


@app.get("/settings")
async def settings_page(request: Request):
    ctx = _nav_context(request, "settings")
    cfg = config.load_config()
    key = cfg.get("minimax_api_key", "")
    ctx["masked_key"] = (key[:6] + "..." + key[-4:]) if len(key) > 12 else ("set" if key else "")
    ctx["config"] = cfg
    return templates.TemplateResponse(request, "settings.html", ctx)


# ---------------------------------------------------------------------------
# API: documents & indexing
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def api_upload(file: Optional[UploadFile] = File(None), text: str = Form(""),
                     title: str = Form("")):
    if not config.get_api_key():
        raise HTTPException(400, "MiniMax API key not configured. Set it on the Settings page.")
    if file is not None and file.filename:
        raw = await file.read()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            content = raw.decode("utf-8", errors="replace")
        filename = file.filename
    elif text.strip():
        content = text.strip()
        filename = title.strip() or "pasted-text.txt"
    else:
        raise HTTPException(400, "Provide a text file or paste some text.")
    if len(content.strip()) < 20:
        raise HTTPException(400, "Document is too short to index (minimum 20 characters).")
    job_id = indexer.start_indexing(filename, content)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def api_job_status(job_id: str):
    job = indexer.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return job


@app.get("/api/documents")
async def api_documents():
    return db.list_documents()


@app.delete("/api/documents/{document_id}")
async def api_delete_document(document_id: int):
    db.delete_document(document_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# API: query
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str


@app.post("/api/query")
async def api_query(body: QueryRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Question is empty.")
    if not config.get_api_key():
        raise HTTPException(400, "MiniMax API key not configured. Set it on the Settings page.")
    try:
        result = retriever.answer_question(question)
    except (EmbeddingError, LLMError) as exc:
        raise HTTPException(502, str(exc))
    retrieval = result["retrieval"]
    return {
        "answer": result["answer"],
        "entities": retrieval["entities"],
        "relationships": [
            {k: v for k, v in r.items() if k != "score"} | ({"score": r.get("score")} if "score" in r else {})
            for r in retrieval["relationships"]
        ],
        "matched_entities": retrieval["matched_entities"],
        "matched_relationships": retrieval["matched_relationships"],
        "chunks": [
            {"id": c["id"], "filename": c["filename"],
             "preview": c["content"][:300] + ("..." if len(c["content"]) > 300 else "")}
            for c in retrieval["chunks"]
        ],
        "seed_ids": retrieval["seed_ids"],
        "hop1_ids": retrieval["hop1_ids"],
        "hop2_ids": retrieval["hop2_ids"],
    }


# ---------------------------------------------------------------------------
# API: graph & settings
# ---------------------------------------------------------------------------

@app.get("/api/graph")
async def api_graph():
    return db.get_full_graph()


class SettingsRequest(BaseModel):
    minimax_api_key: Optional[str] = None
    minimax_group_id: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    top_k_entities: Optional[int] = None
    top_k_relationships: Optional[int] = None
    top_k_chunks: Optional[int] = None


@app.post("/api/settings")
async def api_settings(body: SettingsRequest):
    try:
        data = body.model_dump()
    except AttributeError:
        data = body.dict()
    updates = {k: v for k, v in data.items() if v is not None}
    if "minimax_api_key" in updates and not updates["minimax_api_key"].strip():
        del updates["minimax_api_key"]
    if "minimax_group_id" in updates:
        updates["minimax_group_id"] = updates["minimax_group_id"].strip()
    cfg = config.save_config(updates)
    return {"ok": True, "has_api_key": bool(cfg.get("minimax_api_key"))}


@app.get("/api/stats")
async def api_stats():
    return db.get_stats()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
