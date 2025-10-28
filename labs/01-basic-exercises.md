# Labs — Basic Exercises (mapped to notebooks 02–06)

This file lists five hands-on exercises that accompany the instructor notebooks. Each exercise is short, runnable on CPU, and includes hints. Solutions are in `labs/solutions/exercise_solutions.md`.

## Exercise 1 — Prompt engineering (notebook 02)
- Task: Using `notebooks/02-run-llms-instructor.ipynb`, craft three prompts for a short question: "Explain CRISPR in one sentence". Compare outputs from a small CPU model (`distilgpt2`) and note differences.
- Hint: Try direct instruction, step-by-step, and add constraints (e.g., limit to 10 words).

## Exercise 2 — Embeddings & retrieval (notebook 03)
- Task: Use `labs/data/sample_docs.txt` and compute TF-IDF vectors (sklearn) to implement a simple retriever; query with "What does TP53 do?" and return the top doc.
- Hint: Keep dataset small and use english stop words.

## Exercise 3 — FASTA parsing & QC (notebook 04)
- Task: Parse `labs/data/sample_fasta.fasta`, compute sequence lengths, and flag sequences shorter than 30 nt.
- Hint: Use Biopython's SeqIO or a simple parser.

## Exercise 4 — RAG mini-pipeline (notebook 05)
- Task: Build a tiny KB from `labs/data/sample_docs.txt`, retrieve the most similar doc for "What is BRCA1?", and prompt the model with the retrieved context to generate a short answer.
- Hint: Use embedding fallback (sklearn TF-IDF) if `sentence-transformers` isn't available.

## Exercise 5 — Create training example (notebook 06)
- Task: Add a JSONL example to `labs/data/training_small.jsonl` about BRCA1 (instruction + output) and explain why clear instruction affects outcomes.
- Hint: Keep output short (1-2 sentences).
