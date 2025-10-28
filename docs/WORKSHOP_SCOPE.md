# Workshop Scope — Local AI for Biology

Audience
- Beginners and intermediate practitioners in biology, bioinformatics, and computational biology.
- Basic Python familiarity assumed (reading/writing scripts, installing packages). No deep ML background required.

Duration & Format
- Suggested duration: 1–2 day workshop with short lectures + hands-on labs.
- Delivery: Markdown guides and Jupyter notebooks. Notebooks contain runnable examples and exercises.

Learning Goals
- Understand local AI trade-offs: privacy, cost, reproducibility.
- Be able to set up a local environment (Conda/Mamba, Docker, CUDA drivers).
- Run small/medium LLMs locally (llama.cpp, transformers, Ollama/text-generation-webui)
- Build simple pipelines: tokenization, semantic search (embeddings + FAISS), RAG workflows.
- Apply models to biological tasks: sequence analysis, literature QA, data summarization.

Module Outline
1. Setup & Environment (Course/01-setup-env.md)
2. Running Local LLMs & Model Architectures (notebooks/02-run-llms.ipynb)
3. Embeddings & Semantic Search (notebooks/03-embeddings-faiss.ipynb)
4. RAG for Biology (notebooks/04-rag.ipynb)
5. Working with Biological Sequences (notebooks/05-sequences.ipynb)
6. Fine-tuning / LoRA (labs/06-finetune)

Prerequisites
- Laptop or server with Linux/macOS/Windows (Linux recommended)
- Optional: NVIDIA GPU with recent drivers for acceleration
- Internet access for downloading models (or pre-provide model files for offline workshops)

Software Stack
- Python 3.10+ (managed with Mambaforge)
- Jupyter / JupyterLab
- PyTorch / Transformers / Accelerate
- llama.cpp, text-generation-webui, Ollama (optional)
- FAISS for vector search

Deliverables
- Notebook set for instructors and learners
- Lab exercises with solutions
- Optional Docker images for reproducible environments

Assessment & Exercises
- Short exercises embedded in notebooks (self-graded)
- Instructor solutions in `labs/solutions`

Notes & Assumptions
- This workshop emphasizes practical, reproducible local workflows for biomolecular use-cases.
- For privacy-sensitive data, participants should run everything locally and not use hosted APIs.
