"""Script to index initial documents from storage/rag with automatic metadata generation."""
import asyncio
import sys
import uuid
import re
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters.qdrant.rag_adapter import RAGAdapter
from src.domain.entities.document import DocumentMetadata, DocumentType, NodeMetadata
from src.config import settings


def get_file_type(filename: str) -> DocumentType:
    """Get document type from filename."""
    ext = Path(filename).suffix.lower()
    type_map = {
        ".pdf": DocumentType.PDF,
        ".txt": DocumentType.TXT,
        ".docx": DocumentType.DOCX,
    }
    return type_map.get(ext, DocumentType.TXT)


def extract_year_from_filename(filename: str) -> Optional[int]:
    """Extract year from filename.
    
    Examples:
        "2324.txt" -> 2023 (tahun akademik 2023-2024)
        "2425.txt" -> 2024 (tahun akademik 2024-2025)
        "2023.txt" -> 2023
    """
    # Pattern untuk tahun akademik (2324 = 2023-2024)
    academic_year_pattern = r'^(\d{2})(\d{2})'
    match = re.match(academic_year_pattern, filename)
    if match:
        year1, year2 = match.groups()
        # Jika tahun pertama < 50, berarti 20xx, jika >= 50 berarti 19xx
        if int(year1) < 50:
            return 2000 + int(year1)
        else:
            return 1900 + int(year1)
    
    # Pattern untuk tahun lengkap (2023, 2024, dll)
    full_year_pattern = r'\b(20\d{2})\b'
    match = re.search(full_year_pattern, filename)
    if match:
        return int(match.group(1))
    
    return None


def extract_document_type_from_path(relative_path: Path) -> str:
    """Extract document type (fungsi dokumen) from path.
    
    Mapping kategori folder ke document_type:
    - Kalendar_Akademik -> "kalendar_akademik"
    - Kurikulum -> "kurikulum"
    - Program_Studi -> "program_studi"
    - BIMA -> "organisasi_mahasiswa"
    - FAQ -> "faq"
    - dll
    """
    parts = relative_path.parts
    
    # Mapping folder utama ke document_type
    category_mapping = {
        "Kalendar_Akademik": "kalendar_akademik",
        "Kurikulum": "kurikulum",
        "Program_Studi": "program_studi",
        "Program_STI": "program_sti",
        "BIMA": "organisasi_mahasiswa",
        "FAQ": "faq",
        "FAQ-manual": "faq",
        "Fasilitas Kampus": "fasilitas",
        "Tentang Udinus": "informasi_umum",
        "Data_Dosen": "data_dosen",
        "Beasiswa": "beasiswa",
        "Akreditasi": "akreditasi",
        "AlurAplikasi": "alur_aplikasi",
        "Remidi": "akademik",
        "Semester Antara": "akademik",
        "Pimpinan UDINUS": "informasi_umum",
        "Kuisioner": "kuisioner",
    }
    
    if parts and parts[0] in category_mapping:
        return category_mapping[parts[0]]
    
    # Default berdasarkan folder pertama
    if parts:
        return parts[0].lower().replace(" ", "_")
    
    return "umum"


def extract_section_from_path(relative_path: Path) -> str:
    """Extract section from nested folder structure.
    
    Example:
        BIMA/BEM/BEM_FIB.txt -> "BEM"
        BIMA/BIRO/Biro Animasi/biro_animasi.txt -> "BIRO/Biro Animasi"
    """
    parts = relative_path.parts
    
    # Skip folder pertama (kategori utama) dan nama file
    if len(parts) > 2:
        # Ambil semua folder antara kategori utama dan file
        section_parts = parts[1:-1]
        return "/".join(section_parts)
    
    return ""


def extract_tags_from_path(relative_path: Path) -> List[str]:
    """Extract tags from folder structure.
    
    Tags akan diambil dari semua level folder (kecuali nama file).
    """
    parts = relative_path.parts
    tags = []
    
    # Tambahkan semua folder sebagai tags
    for part in parts[:-1]:  # Exclude filename
        # Normalize tag name
        tag = part.replace("_", " ").replace("-", " ").title()
        tags.append(tag)
    
    return tags


def extract_metadata_from_path(
    file_path: Path, 
    base_path: Path
) -> Tuple[str, str, str, List[str], Optional[int]]:
    """Extract all metadata from file path.
    
    Returns:
        Tuple of (category, document_type, section, tags, year)
    """
    relative_path = file_path.relative_to(base_path)
    parts = relative_path.parts
    
    # Category = folder utama
    category = parts[0] if parts else "general"
    
    # Document type
    document_type = extract_document_type_from_path(relative_path)
    
    # Section = subfolder structure
    section = extract_section_from_path(relative_path)
    
    # Tags = semua folder dalam path
    tags = extract_tags_from_path(relative_path)
    
    # Year dari filename
    year = extract_year_from_filename(file_path.name)
    
    return category, document_type, section, tags, year


async def index_initial_documents(
    source_dir: Path = None,
    skip_existing: bool = True,
    dry_run: bool = False,
):
    """Index all initial documents from storage/rag with category metadata.
    
    Args:
        source_dir: Source directory (default: storage/rag)
        skip_existing: Skip documents already indexed
        dry_run: If True, only show what would be indexed
    """
    if source_dir is None:
        source_dir = Path(settings.rag_storage_path).parent  # storage/rag
    
    print(f"Initializing RAG system...")
    print(f"Source directory: {source_dir}")
    
    # Initialize RAG adapter
    rag_adapter = RAGAdapter()
    await rag_adapter.initialize()
    
    # Get already indexed document IDs (if skip_existing)
    indexed_ids = set()
    if skip_existing:
        indexed_ids = set(await rag_adapter.get_all_document_ids())
        print(f"Found {len(indexed_ids)} already indexed documents")
    
    # Supported extensions
    extensions = [".pdf", ".txt", ".docx"]
    
    # Find all documents (exclude system folders)
    documents = []
    for ext in extensions:
        for doc_path in source_dir.rglob(f"*{ext}"):
            # Skip system folders
            if "documents" in doc_path.parts or "metadata" in doc_path.parts:
                continue
            documents.append(doc_path)
    
    print(f"\nFound {len(documents)} documents to index")
    
    if dry_run:
        print("\n[DRY RUN] Would index:")
        for doc_path in documents[:10]:  # Show first 10
            category, doc_type, section, tags, year = extract_metadata_from_path(
                doc_path, source_dir
            )
            print(f"  - {doc_path.relative_to(source_dir)}")
            print(f"    Category: {category}")
            print(f"    Document Type: {doc_type}")
            print(f"    Section: {section or '(none)'}")
            print(f"    Tags: {', '.join(tags) if tags else '(none)'}")
            print(f"    Year: {year or '(none)'}")
        if len(documents) > 10:
            print(f"  ... and {len(documents) - 10} more")
        return
    
    # Index each document
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for i, doc_path in enumerate(documents, 1):
        # Extract all metadata from path
        category, document_type, section, tags, year = extract_metadata_from_path(
            doc_path, source_dir
        )
        
        # Generate document ID (use category + filename for consistency)
        # Normalize category and filename for ID
        category_normalized = category.lower().replace(" ", "_").replace("-", "_")
        filename_normalized = doc_path.stem.lower().replace(" ", "_").replace("-", "_")
        doc_id = f"{category_normalized}_{filename_normalized}_{uuid.uuid4().hex[:8]}"
        
        # Skip if already indexed
        if skip_existing and doc_id in indexed_ids:
            print(f"[{i}/{len(documents)}] ⏭️  Skipped (already indexed): {doc_path.name}")
            skipped_count += 1
            continue
        
        print(f"\n[{i}/{len(documents)}] Processing: {doc_path.relative_to(source_dir)}")
        print(f"  Category: {category}")
        print(f"  Document Type: {document_type}")
        if section:
            print(f"  Section: {section}")
        if tags:
            print(f"  Tags: {', '.join(tags)}")
        if year:
            print(f"  Year: {year}")
        
        try:
            # Create document metadata
            metadata = DocumentMetadata(
                filename=doc_path.name,
                file_type=get_file_type(doc_path.name),
                file_size=doc_path.stat().st_size,
                uploaded_at=datetime.now(),
            )
            
            # Create node metadata template with extracted metadata
            # Tags akan digunakan untuk semua chunks dari dokumen ini
            # Position dan chunk_id akan di-generate otomatis oleh rag_adapter
            node_metadata_template = NodeMetadata(
                doc_id=doc_id,  # Akan digunakan untuk semua chunks
                chunk_id="",  # Akan di-generate otomatis per chunk (format: {doc_id}_{idx:02d})
                tags=tags if tags else [category],  # Tags dari folder structure atau fallback ke category
                source=category,  # Source = kategori utama (folder pertama)
                document_type=document_type,  # Document type dari mapping kategori
                section=section,  # Section dari subfolder structure
                language="id",  # Default bahasa Indonesia
                year=year,  # Year yang di-extract dari filename (jika ada)
                position=0,  # Akan di-generate otomatis per chunk (0, 1, 2, ...)
            )
            
            # Index document
            num_chunks = await rag_adapter.add_document(
                document_id=doc_id,
                file_path=doc_path,
                metadata=metadata,
                node_metadata_template=node_metadata_template,
            )
            
            print(f"  ✓ Success: {num_chunks} chunks created")
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Indexing complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Total: {len(documents)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Index initial documents to RAG")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Source directory (default: storage/rag)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-index existing documents",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be indexed without actually indexing",
    )
    
    args = parser.parse_args()
    
    asyncio.run(
        index_initial_documents(
            source_dir=args.source_dir,
            skip_existing=not args.no_skip_existing,
            dry_run=args.dry_run,
        )
    )