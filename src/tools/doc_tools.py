"""
Forensic tools for the DocAnalyst detective.

Implements a "RAG-lite" approach: the PDF is chunked into passages,
and the agent can query specific topics without dumping the entire
document into context.

Uses PyMuPDF (fitz) for PDF parsing — install with: pip install pymupdf
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# PDF Text Extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Extract all text from a PDF file using PyMuPDF.
    Returns the full text as a single string, or None on failure.
    """
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n".join(pages)
    except ImportError:
        # Fallback: try pdfplumber
        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                return "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
        except ImportError:
            return None
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


def extract_images_from_pdf(pdf_path: str, output_dir: str) -> List[str]:
    """
    Extract images from a PDF and save them as PNG files.
    Returns list of saved image paths.
    """
    saved: List[str] = []
    try:
        import fitz

        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                ext = base_image.get("ext", "png")
                out_path = Path(output_dir) / f"page{page_num}_img{img_index}.{ext}"
                out_path.write_bytes(img_bytes)
                saved.append(str(out_path))
        doc.close()
    except Exception as exc:  # noqa: BLE001
        print(f"Image extraction failed: {exc}")
    return saved


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks of approximately chunk_size words.
    Overlap helps avoid cutting off context mid-sentence.
    """
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# RAG-lite Query
# ---------------------------------------------------------------------------

def query_pdf(text: str, query: str, top_k: int = 3) -> List[str]:
    """
    Simple keyword-based retrieval over chunked PDF text.
    Returns the top_k most relevant chunks for a query.

    For the interim submission this uses TF-IDF-style keyword scoring.
    In the final version this can be replaced with an embedding-based search.
    """
    chunks = chunk_text(text)
    query_tokens = set(re.findall(r"\w+", query.lower()))

    scored: List[Tuple[float, str]] = []
    for chunk in chunks:
        chunk_tokens = re.findall(r"\w+", chunk.lower())
        chunk_token_set = set(chunk_tokens)
        if not chunk_tokens:
            continue
        overlap = len(query_tokens & chunk_token_set)
        score = overlap / len(chunk_tokens)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


# ---------------------------------------------------------------------------
# File Path Extraction (for cross-referencing with RepoInvestigator)
# ---------------------------------------------------------------------------

def extract_file_paths_from_text(text: str) -> List[str]:
    """
    Extract any file path references from text.
    Matches patterns like: src/tools/repo_tools.py, ./graph.py, nodes/judges.py
    """
    pattern = r"(?:src/|\./)?\w[\w/]*\.(?:py|md|json|toml|txt|yaml|yml)"
    return list(set(re.findall(pattern, text)))


# ---------------------------------------------------------------------------
# Keyword Depth Analysis
# ---------------------------------------------------------------------------

DEEP_CONCEPTS = [
    "Dialectical Synthesis",
    "Fan-In",
    "Fan-Out",
    "Metacognition",
    "State Synchronization",
    "parallel",
    "LangGraph",
    "StateGraph",
    "reducer",
    "operator.add",
    "operator.ior",
]


def analyze_concept_depth(text: str) -> Dict[str, Dict]:
    """
    For each key concept, check whether it appears in the PDF and
    whether it's accompanied by a substantive explanation (>50 words context)
    or just a buzzword drop.

    Returns: { concept: { found: bool, depth: "deep" | "shallow" | "absent", excerpt: str } }
    """
    results: Dict[str, Dict] = {}
    lower = text.lower()

    for concept in DEEP_CONCEPTS:
        idx = lower.find(concept.lower())
        if idx == -1:
            results[concept] = {"found": False, "depth": "absent", "excerpt": ""}
            continue

        # Extract surrounding context (200 chars each side)
        start = max(0, idx - 200)
        end = min(len(text), idx + 200)
        excerpt = text[start:end].strip()

        # Count words in excerpt as a proxy for explanation depth
        word_count = len(excerpt.split())
        depth = "deep" if word_count > 50 else "shallow"

        results[concept] = {
            "found": True,
            "depth": depth,
            "excerpt": excerpt,
        }

    return results


# ---------------------------------------------------------------------------
# High-Level Entry Point (used by DocAnalyst node)
# ---------------------------------------------------------------------------

def ingest_pdf(pdf_path: str) -> Dict:
    """
    Full ingestion pipeline for a PDF report.
    Returns a dict with:
        - full_text
        - chunks
        - file_paths_mentioned
        - concept_depth_analysis
    """
    if not Path(pdf_path).exists():
        return {"error": f"PDF not found: {pdf_path}"}

    text = extract_text_from_pdf(pdf_path)
    if text is None:
        return {"error": "Could not extract text — install pymupdf or pdfplumber"}

    if text.startswith("ERROR:"):
        return {"error": text}

    chunks = chunk_text(text)
    file_paths = extract_file_paths_from_text(text)
    concept_depth = analyze_concept_depth(text)

    return {
        "pdf_path": pdf_path,
        "full_text": text,
        "chunks": chunks,
        "chunk_count": len(chunks),
        "file_paths_mentioned": file_paths,
        "concept_depth": concept_depth,
    }
