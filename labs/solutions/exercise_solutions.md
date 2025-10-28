# Exercise Solutions (instructor answers)

## Exercise 1 — Prompt engineering (notebook 02)
Prompts and example outputs (answers will vary by model):

1) Direct: "Explain CRISPR in one sentence."
- Expected: "CRISPR is a genome editing tool that uses guide RNA and Cas proteins to target DNA."

2) Step-by-step: "In one sentence, explain CRISPR to a beginner, using simple language."
- Expected: "CRISPR is a biological tool that lets scientists change DNA by guiding molecular scissors to a target."

3) Constraint: "Explain CRISPR in one sentence using at most 10 words."
- Expected: "CRISPR: guide RNA directs enzymes to edit DNA." (or similar concise reply)

## Exercise 2 — Embeddings & retrieval (notebook 03)
Solution outline (TF-IDF):
- Load `labs/data/sample_docs.txt`.
- Use sklearn.feature_extraction.text.TfidfVectorizer(stop_words='english') to vectorize.
- Vectorize query and compute cosine similarity with doc vectors.
- Return the highest scoring doc (expected: TP53 or CRISPR doc depending on query).

## Exercise 3 — FASTA parsing & QC (notebook 04)
Solution using Biopython:

from Bio import SeqIO
for rec in SeqIO.parse('labs/data/sample_fasta.fasta', 'fasta'):
    print(rec.id, len(rec.seq))
    if len(rec.seq) < 30:
        print('FLAG SHORT SEQ', rec.id)

Expected: all sequences are >30 in the sample, but flagging logic should work.

## Exercise 4 — RAG mini-pipeline (notebook 05)
Solution outline:
- Build document corpus from `labs/data/sample_docs.txt`.
- Use TF-IDF to find top doc for query "What is BRCA1?" (expected: BRCA1 summary doc).
- Compose prompt: "Based on the following context: <top_doc>\nQuestion: What is BRCA1?\nAnswer:" and call a small model for completion.

## Exercise 5 — Create training example (notebook 06)
Sample JSONL object to append to `labs/data/training_small.jsonl`:
{
  "instruction": "Summarize BRCA1 in one sentence.",
  "input": "",
  "output": "BRCA1 is a gene involved in DNA repair and is associated with hereditary breast and ovarian cancer risk."
}

Instructor note: Explain that precise instructions reduce ambiguity and improve the model's responses, especially with small datasets.
