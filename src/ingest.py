from dotenv import load_dotenv
load_dotenv() #loading the env varables

import os
from glob import glob

from langchain_community.document_loaders import TextLoader, NotebookLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

DATA_DIR = "data/raw"
CHROMA_DIR = "chroma_db"
COLLECTION = "openai_cookbook"

def load_docs():
    docs = []

    
    # Markdown files
    for path in glob(os.path.join(DATA_DIR, "*.md")):
        loaded = TextLoader(path, encoding="utf-8").load()
        for d in loaded:
            d.metadata["source_file"] = os.path.basename(path)
        docs.extend(loaded)

    # Notebooks
    for path in glob(os.path.join(DATA_DIR, "*.ipynb")):
        loaded = NotebookLoader(path).load()
        for d in loaded:
            d.metadata["source_file"] = os.path.basename(path)
        docs.extend(loaded)

    return docs

def main():
    docs = load_docs()
    
    print(f"Docs loaded: {len(docs)}")

    for i, d in enumerate(docs[:3]):
        print(f"\n--- Doc {i} ---")
        print("Metadata:", d.metadata)
        print("Content preview:", repr(d.page_content[:300]))
    # text = """
    # Your long document text goes here...
    # Add multiple paragraphs to test chunking behavior.
    # """

    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=10,
    #     chunk_overlap=1
    # )

    # chunks = splitter.split_text(text)

    # for i, chunk in enumerate(chunks):
    # print(f"\n--- Chunk {i} ---")
    # print(chunk)
    # print(f"Length: {len(chunk)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    # remove empty / whitespace-only chunks
    chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
    chunks = chunks[:20] ##limiting for rate limit control
    print(f"Non-empty chunks: {len(chunks)}")

    if not chunks:
        raise RuntimeError("All chunks were empty after filtering. Check loaders / file contents.")


    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(collection_name=COLLECTION,
                embedding_function=embeddings,
                persist_directory=CHROMA_DIR)
    db.add_documents(chunks)

    BATCH = 100
    for i in range(0, len(chunks), BATCH):
        db.add_documents(chunks[i:i+BATCH])
        print(f"Upserted {min(i+BATCH, len(chunks))}/{len(chunks)}")

    print(f"Stored vectors in: {CHROMA_DIR}/ (collection: {COLLECTION})")

if __name__ == "__main__":
    main()