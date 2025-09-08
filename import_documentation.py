#!/usr/bin/env python3
"""
Script to import documentation into AI Memory Service
"""

import os
import asyncio
import aiohttp
import json
from pathlib import Path
from typing import List, Dict
import hashlib
import mimetypes

# Configuration
API_HOST = os.getenv("MEMORY_API_HOST", "localhost")
API_PORT = os.getenv("MEMORY_API_PORT", "8080")
API_BASE = f"http://{API_HOST}:{API_PORT}"
DOCS_PATH = r"C:\Users\1\Documents\Документация"

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    '.txt', '.md', '.rst', '.doc', '.docx', 
    '.pdf', '.html', '.htm', '.xml', '.json',
    '.py', '.js', '.ts', '.java', '.cpp', '.c',
    '.h', '.hpp', '.cs', '.go', '.rs', '.rb',
    '.php', '.swift', '.kt', '.scala', '.r',
    '.yaml', '.yml', '.toml', '.ini', '.cfg',
    '.sql', '.sh', '.ps1', '.bat', '.cmd'
}

def generate_memory_id(content: str) -> str:
    """Generate unique ID for memory"""
    return hashlib.sha256(content.encode()).hexdigest()[:16]

def extract_metadata(file_path: Path) -> Dict:
    """Extract metadata from file"""
    stat = file_path.stat()
    mime_type, _ = mimetypes.guess_type(str(file_path))
    
    return {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "file_size": str(stat.st_size),
        "mime_type": mime_type or "text/plain",
        "modified": str(stat.st_mtime),
        "extension": file_path.suffix.lower()
    }

def chunk_content(content: str, max_size: int = 4000) -> List[str]:
    """Split content into chunks for processing"""
    if len(content) <= max_size:
        return [content]
    
    chunks = []
    words = content.split()
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_size = len(word) + 1  # +1 for space
        if current_size + word_size > max_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

async def read_file_content(file_path: Path) -> str:
    """Read file content with proper encoding handling"""
    encodings = ['utf-8', 'cp1251', 'cp866', 'latin-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    # If all encodings fail, try binary mode and decode with errors ignored
    try:
        with open(file_path, 'rb') as f:
            return f.read().decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"⚠️  Could not read {file_path}: {e}")
        return ""

async def store_memory(session: aiohttp.ClientSession, memory_data: Dict) -> bool:
    """Store a single memory in the service"""
    try:
        async with session.post(
            f"{API_BASE}/api/memories",
            json=memory_data,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status in [200, 201]:
                return True
            else:
                error_text = await response.text()
                print(f"❌ Failed to store memory: {response.status} - {error_text[:100]}")
                return False
    except Exception as e:
        print(f"❌ Error storing memory: {e}")
        return False

async def process_file(session: aiohttp.ClientSession, file_path: Path) -> int:
    """Process a single file and store as memories"""
    print(f"📄 Processing: {file_path.name}")
    
    # Read file content
    content = await read_file_content(file_path)
    if not content:
        return 0
    
    # Extract metadata
    metadata = extract_metadata(file_path)
    
    # Create context path from file path
    rel_path = str(file_path).replace(DOCS_PATH, "").strip("\\").strip("/")
    context_path = f"docs/{rel_path.replace(os.sep, '/')}"
    
    # Chunk content if needed
    chunks = chunk_content(content)
    stored_count = 0
    
    for i, chunk in enumerate(chunks):
        # Prepare memory data
        memory_data = {
            "content": chunk,
            "context_path": context_path,
            "importance": 0.7,  # Documentation is important
            "metadata": {
                **metadata,
                "chunk_index": str(i),
                "total_chunks": str(len(chunks)),
                "source": "documentation_import"
            }
        }
        
        # Store memory
        if await store_memory(session, memory_data):
            stored_count += 1
    
    return stored_count

async def scan_directory(path: Path) -> List[Path]:
    """Recursively scan directory for supported files"""
    files = []
    
    try:
        for item in path.rglob("*"):
            if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
                # Skip very large files (>10MB)
                if item.stat().st_size < 10 * 1024 * 1024:
                    files.append(item)
    except Exception as e:
        print(f"⚠️  Error scanning {path}: {e}")
    
    return files

async def main():
    print("=" * 60)
    print("📚 Documentation Import Tool for AI Memory Service")
    print("=" * 60)
    print()
    
    docs_path = Path(DOCS_PATH)
    
    if not docs_path.exists():
        print(f"❌ Documentation path does not exist: {DOCS_PATH}")
        return
    
    print(f"📂 Scanning: {DOCS_PATH}")
    files = await scan_directory(docs_path)
    
    if not files:
        print("❌ No supported files found")
        return
    
    print(f"✅ Found {len(files)} files to import")
    print()
    
    # Limit to first 100 files for initial import
    if len(files) > 100:
        print(f"⚠️  Limiting to first 100 files")
        files = files[:100]
    
    async with aiohttp.ClientSession() as session:
        # First, check if API is available
        try:
            async with session.get(f"{API_BASE}/health") as response:
                if response.status != 200:
                    print(f"⚠️  API health check failed: {response.status}")
        except Exception as e:
            print(f"❌ Cannot connect to API at {API_BASE}: {e}")
            print("   Make sure the memory service is running")
            return
        
        print("🚀 Starting import...")
        print("-" * 40)
        
        total_memories = 0
        processed_files = 0
        
        for file_path in files:
            try:
                memories_stored = await process_file(session, file_path)
                if memories_stored > 0:
                    total_memories += memories_stored
                    processed_files += 1
                    print(f"   ✅ Stored {memories_stored} memory chunks")
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            # Small delay to avoid overwhelming the API
            await asyncio.sleep(0.1)
        
        print("-" * 40)
        print()
        print("📊 Import Summary:")
        print(f"   Files processed: {processed_files}/{len(files)}")
        print(f"   Total memories stored: {total_memories}")
        print(f"   Average memories per file: {total_memories/processed_files:.1f}" if processed_files > 0 else "")
    
    print()
    print("=" * 60)
    print("✅ Import Complete")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())