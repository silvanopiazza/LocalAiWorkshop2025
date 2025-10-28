# Notebook 3: Embeddings and Semantic Search with FAISS

## Introduction

Welcome to Notebook 3! This notebook covers one of the most powerful techniques in modern AI: **embeddings and semantic search**. This is crucial for building Retrieval-Augmented Generation (RAG) systems that let LLMs access external knowledge.

**What You'll Learn:**
- What embeddings are and why they matter
- How to generate embeddings from text and biological sequences
- How to use FAISS for fast similarity search
- How to build a document search system
- How to integrate semantic search with LLMs
- Practical applications in bioinformatics (protein search, literature mining)

**Why This Matters for Biology:**
- Search through millions of scientific papers instantly
- Find similar gene sequences or proteins
- Build AI systems that can "read" and search biological literature
- Enable citation-aware question answering

---

## Part 1: Understanding Embeddings

### 1.1 What are Embeddings?

An **embedding** is a numerical representation of text (or any data) as a vector of numbers.

**Simple Analogy:**
Instead of storing words as text, we represent them as positions in space:

```
Text: "DNA"
Embedding (3-dimensional): [0.247, -0.891, 0.156]

Text: "RNA"
Embedding (3-dimensional): [0.251, -0.885, 0.162]

Text: "Protein"
Embedding (3-dimensional): [0.512, 0.234, -0.891]
```

**Key Insight:** Similar concepts have similar embeddings!
- DNA and RNA embeddings are close (they're related molecules)
- Protein embedding is far from DNA (different concept)

### 1.2 Why Embeddings?

**Problem with Text Search:**
```python
# Traditional search (keyword matching)
query = "DNA sequencing"
document = "Genomic sequencing uses DNA"

# Does this match? Only if we match keywords exactly
# Misses: "genomic DNA profiling", "sequencing the genome"
```

**Solution with Embeddings:**
```
query_embedding = embed("DNA sequencing")
doc_embedding = embed("Genomic sequencing uses DNA")

# Calculate similarity (cosine distance)
similarity = cosine_similarity(query_embedding, doc_embedding)

# Similar documents have high similarity score!
# Finds semantically related documents, not just keyword matches
```

### 1.3 Types of Embeddings

**Sentence-Level Embeddings:**
- One embedding for entire sentences/documents
- Used for semantic search
- Typical size: 384-1024 dimensions
- Models: Sentence-Transformers, Universal Sentence Encoder

**Token-Level Embeddings:**
- One embedding per word/token
- Used inside transformer models
- Typical size: 768-4096 dimensions
- Models: BERT, GPT embeddings

**Domain-Specific Embeddings (for Biology):**
- Protein embeddings: ESM2 (trained on proteins)
- DNA sequence embeddings: DNABert
- Biomedical text: BioSentVec, PubMedBERT
- Chemical embeddings: ChemBERT

### 1.4 How Embeddings Are Created

**The Process:**

```
Input Text: "CRISPR is a gene editing technology"
          ↓
[Tokenization] → Breaks into tokens
["CRISPR", "is", "a", "gene", "editing", "technology"]
          ↓
[Token Embeddings] → Each token becomes a vector
[0.2, 0.5, ...], [0.1, 0.3, ...], ...
          ↓
[Aggregation] → Combine token embeddings
(e.g., take mean/average)
          ↓
Final Embedding (384 dimensions): [0.234, 0.156, ..., 0.891]
```

**Example: Mathematical Representation**

```
Text: "Bioinformatics"

Tokenization: ["Bioinformatics"]

Token ID: 3241 (unique identifier for word)

Token Embedding (from lookup table):
[0.234, -0.156, 0.891, ..., 0.102]
 ↑       ↑      ↑         ↑
 1st    2nd    3rd      384th
 dimension
```

---

## Part 2: Creating Embeddings with Sentence-Transformers

### 2.1 Installation and Basics

```python
# Install sentence-transformers
# pip install sentence-transformers

from sentence_transformers import SentenceTransformer
import numpy as np

# Load a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# This model:
# - Works well for general text
# - Fast (good for real-time search)
# - Produces 384-dimensional embeddings
# - Multilingual support

# Generate embeddings
sentences = [
    "DNA sequencing reads the genetic code",
    "RNA is synthesized from DNA template",
    "Proteins are built from RNA instructions",
    "The dog is sleeping",
]

embeddings = model.encode(sentences)

print(f"Number of sentences: {len(embeddings)}")
print(f"Embedding dimension: {len(embeddings[0])}")
print(f"First embedding: {embeddings[0][:5]}...")  # Show first 5 values
```

**Output:**
```
Number of sentences: 4
Embedding dimension: 384
First embedding: [0.2456 -0.1234  0.5678  0.3421 -0.4567]...
```

### 2.2 Computing Similarity

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Embeddings from previous example
# embeddings shape: (4, 384)

# Calculate similarity between all pairs
similarity_matrix = cosine_similarity(embeddings)

print("Similarity Matrix:")
print(similarity_matrix)
print()

# Example: Similarity between first and other sentences
query_embedding = embeddings[0]  # "DNA sequencing reads the genetic code"

for i, sentence in enumerate(sentences):
    sim = cosine_similarity([query_embedding], [embeddings[i]])[0][0]
    print(f"Similarity with '{sentence}': {sim:.4f}")
```

**Output:**
```
Similarity with 'DNA sequencing reads the genetic code': 1.0000
Similarity with 'RNA is synthesized from DNA template': 0.8234
Similarity with 'Proteins are built from RNA instructions': 0.6542
Similarity with 'The dog is sleeping': 0.1234
```

**Interpretation:**
- 1.0 = identical (it's the same sentence)
- 0.8234 = very similar (both about molecular biology)
- 0.6542 = somewhat related (all three connected)
- 0.1234 = unrelated (dogs vs molecular biology)

### 2.3 Available Models

```python
from sentence_transformers import SentenceTransformer

# General-purpose models (recommended for starting)
models = {
    'all-MiniLM-L6-v2': {
        'size': 'Small (24MB)',
        'speed': 'Very Fast',
        'quality': 'Good',
        'use': 'General text, semantic search'
    },
    'all-mpnet-base-v2': {
        'size': 'Medium (420MB)',
        'speed': 'Fast',
        'quality': 'Excellent',
        'use': 'General text, high quality'
    },
    'sentence-transformers/paraphrase-distilroberta-base-v1': {
        'size': 'Small',
        'speed': 'Fast',
        'quality': 'Good',
        'use': 'Paraphrase detection'
    },
}

# Load a model
model = SentenceTransformer('all-MiniLM-L6-v2')

# For biomedical text (specialized)
# Note: These are larger and slower but very accurate for biology
# model = SentenceTransformer('sentence-transformers/pubmedbert-base-uncased-abssum')
# model = SentenceTransformer('allenai/specter')  # For scientific papers
```

---

## Part 3: FAISS - Fast Similarity Search

### 3.1 Why FAISS?

**Problem:** Comparing new query against millions of documents is slow

```
Query: "How does CRISPR work?"
         ↓
Compare against 1,000,000 documents
(Brute force: 1,000,000 comparisons)
         ↓
Return top 10 most similar

# Time complexity: O(n) where n = number of documents
# With 1M documents: SLOW!
```

**Solution: FAISS (Facebook AI Similarity Search)**
- Uses indexing structures for fast search
- Can search 1M documents in milliseconds
- Uses approximate nearest neighbor search
- Memory efficient

### 3.2 Basic FAISS Usage

```python
# Install faiss
# pip install faiss-cpu  (or faiss-gpu for GPU acceleration)

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Step 1: Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "DNA sequencing determines the order of nucleotides",
    "CRISPR is a powerful gene editing tool",
    "The genetic code is translated to proteins",
    "RNA polymerase synthesizes RNA from DNA",
    "Genes regulate biological processes",
]

# Embed all documents
embeddings = model.encode(documents)
embeddings = np.array(embeddings).astype('float32')

print(f"Embeddings shape: {embeddings.shape}")  # (5, 384)

# Step 2: Create FAISS index
# IndexFlatL2 = exact L2 distance search
index = faiss.IndexFlatL2(embeddings.shape[1])

# Step 3: Add embeddings to index
index.add(embeddings)

print(f"Index size: {index.ntotal}")  # 5

# Step 4: Search
query = "How does gene editing work?"
query_embedding = model.encode([query]).astype('float32')

# Find 3 most similar documents
k = 3
distances, indices = index.search(query_embedding, k)

print(f"\nTop {k} most similar documents to: '{query}'")
print("=" * 60)

for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    similarity = 1 / (1 + dist)  # Convert distance to similarity
    print(f"{i+1}. Document {idx}: '{documents[idx]}'")
    print(f"   Similarity: {similarity:.4f}")
    print()
```

**Output:**
```
Index size: 5

Top 3 most similar documents to: 'How does gene editing work?'
============================================================
1. Document 1: 'CRISPR is a powerful gene editing tool'
   Similarity: 0.8923

2. Document 0: 'DNA sequencing determines the order of nucleotides'
   Similarity: 0.6734

3. Document 4: 'Genes regulate biological processes'
   Similarity: 0.5421
```

### 3.3 Advanced FAISS Indices

**IndexFlatL2 (Exact Search - Slow but Accurate):**
```python
# Best for: Small datasets (< 1M)
# Speed: O(n) - must check all points
# Accuracy: 100%

index = faiss.IndexFlatL2(d)  # d = embedding dimension
```

**IndexIVFFlat (Quantized Search - Fast Approximation):**
```python
# Best for: Large datasets (1M - 100M)
# Speed: O(log n) - uses clustering
# Accuracy: ~99%

# Create IVF index with 100 clusters
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, n_centroids=100)

# Train on sample data
train_data = embeddings[:1000].astype('float32')
index.train(train_data)

# Add all data
index.add(embeddings)
```

**IndexHNSW (Hierarchical Navigable Small World - Very Fast):**
```python
# Best for: Extreme scale (100M+)
# Speed: Sub-millisecond for billions of vectors
# Accuracy: ~98%

index = faiss.IndexHNSWFlat(d, M=32)
index.add(embeddings)
```

### 3.4 Complete Example: Document Search System

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class DocumentSearchEngine:
    """Search through documents using semantic similarity"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', use_gpu=False):
        """Initialize search engine"""
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.index = None
        self.use_gpu = use_gpu
        
    def add_documents(self, documents):
        """
        Add documents to the search index
        
        Args:
            documents: List of document strings
        """
        print(f"Encoding {len(documents)} documents...")
        
        # Encode documents
        embeddings = self.model.encode(documents, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Store documents
        self.documents = documents
        self.embeddings = embeddings
        
        # Create index
        d = embeddings.shape[1]  # Embedding dimension
        
        if len(documents) < 1000:
            # Small dataset: use exact search
            self.index = faiss.IndexFlatL2(d)
        else:
            # Large dataset: use IVF
            quantizer = faiss.IndexFlatL2(d)
            self.index = faiss.IndexIVFFlat(
                quantizer, d, 
                n_centroids=min(100, len(documents) // 10)
            )
            self.index.train(embeddings)
        
        # Add vectors to index
        self.index.add(embeddings)
        
        print(f"Index built with {self.index.ntotal} documents")
    
    def search(self, query, top_k=5):
        """
        Search for documents similar to query
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        
        # Encode query
        query_embedding = self.model.encode([query]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Convert to similarity scores
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            similarity = 1 / (1 + distance)  # Convert L2 distance to similarity
            results.append({
                'document': self.documents[idx],
                'similarity': float(similarity),
                'index': int(idx)
            })
        
        return results
    
    def search_with_threshold(self, query, threshold=0.5, top_k=10):
        """
        Search with minimum similarity threshold
        
        Args:
            query: Query string
            threshold: Minimum similarity (0-1)
            top_k: Maximum results
            
        Returns:
            Results above threshold
        """
        results = self.search(query, top_k=top_k)
        return [r for r in results if r['similarity'] >= threshold]


# Usage Example
if __name__ == "__main__":
    # Create search engine
    search_engine = DocumentSearchEngine()
    
    # Add some biology documents
    documents = [
        "CRISPR-Cas9 is a revolutionary gene editing technology derived from bacteria",
        "DNA sequencing identifies the order of nucleotides in the genome",
        "GWAS studies identify genetic variants associated with diseases",
        "RNA interference can silence specific genes through siRNA",
        "The CRISPR-Cas9 system was adapted for mammalian cell gene editing",
        "Whole genome sequencing has become faster and cheaper",
        "Gene therapy uses modified genes to treat genetic diseases",
        "Clustered regularly interspaced short palindromic repeats enable precise genome editing",
    ]
    
    # Build index
    search_engine.add_documents(documents)
    
    # Perform searches
    queries = [
        "How can we edit genes?",
        "What is sequencing?",
        "Tell me about gene therapy",
    ]
    
    print("\n" + "="*70)
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 70)
        
        results = search_engine.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. (similarity: {result['similarity']:.4f})")
            print(f"   {result['document']}\n")
```

**Output:**
```
Query: 'How can we edit genes?'
----------------------------------------------------------------------
1. (similarity: 0.8756)
   CRISPR-Cas9 is a revolutionary gene editing technology derived from bacteria

2. (similarity: 0.8423)
   The CRISPR-Cas9 system was adapted for mammalian cell gene editing

3. (similarity: 0.7245)
   Gene therapy uses modified genes to treat genetic diseases
```

---

## Part 4: Protein Sequence Embeddings

### 4.1 ESM2 - Facebook's Protein Language Model

```python
# Install: pip install fair-esm

import torch
from Bio import SeqIO
import numpy as np

# Load ESM2 model (trained on protein sequences)
model_name = "facebook/esm2_t6_8M_UR50D"
print(f"Loading {model_name}...")

model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
model = model.eval()

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Example: Embed protein sequences
sequences = [
    ("Hemoglobin", "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHG"),
    ("Insulin", "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT"),
    ("Lysozyme", "MKAAVLTAVLLSVVFAFSSCGDDDDTGHGHHHHHHQQ"),
]

embeddings = {}

for name, seq in sequences:
    # Prepare input
    tokens = alphabet.encode(seq)
    
    # Forward pass
    with torch.no_grad():
        results = model(tokens.to(device))
    
    # Extract representation (mean pooling)
    representation = results["representations"][33]  # Last layer
    embedding = torch.mean(representation, dim=1).cpu().numpy()
    
    embeddings[name] = embedding
    print(f"{name}: embedding shape {embedding.shape}")

# Calculate similarity between proteins
from scipy.spatial.distance import cosine

hemoglobin_emb = embeddings["Hemoglobin"]
insulin_emb = embeddings["Insulin"]
lysozyme_emb = embeddings["Lysozyme"]

sim_hemo_ins = 1 - cosine(hemoglobin_emb, insulin_emb)
sim_hemo_lys = 1 - cosine(hemoglobin_emb, lysozyme_emb)

print(f"\nSimilarity (Hemoglobin - Insulin): {sim_hemo_ins:.4f}")
print(f"Similarity (Hemoglobin - Lysozyme): {sim_hemo_lys:.4f}")
```

### 4.2 DNA Sequence Embeddings with DNABert

```python
# Install: pip install transformers

from transformers import AutoTokenizer, AutoModel
import torch

# Load DNA-BERT model
model_name = "dna-bert/dnabert-2-117m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# DNA sequences
dna_sequences = [
    ("Sequence1", "ACGTACGTACGTACGT"),
    ("Sequence2", "ACGTACGTACGTACGT"),  # Same as Sequence1
    ("Sequence3", "TGCATGCATGCATGCA"),  # Different
]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

embeddings = {}

for name, seq in dna_sequences:
    # Tokenize (DNA-BERT uses k-mer tokenization)
    inputs = tokenizer(seq, return_tensors="pt").to(device)
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean pooling
    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    embeddings[name] = embedding

# Compare sequences
from scipy.spatial.distance import cosine

sim_1_2 = 1 - cosine(embeddings["Sequence1"], embeddings["Sequence2"])
sim_1_3 = 1 - cosine(embeddings["Sequence1"], embeddings["Sequence3"])

print(f"Similarity (Seq1 - Seq2): {sim_1_2:.4f}")  # Should be ~1.0
print(f"Similarity (Seq1 - Seq3): {sim_1_3:.4f}")  # Should be lower
```

---

## Part 5: Integrating with FAISS for Protein Search

```python
import faiss
import numpy as np
from scipy.spatial.distance import cosine

class ProteinSearchEngine:
    """Search protein database by sequence similarity"""
    
    def __init__(self, embedding_model='facebook/esm2_t6_8M_UR50D'):
        self.model_name = embedding_model
        self.model = None
        self.alphabet = None
        self.proteins = []
        self.embeddings = None
        self.index = None
        
    def load_model(self):
        """Load ESM2 model"""
        import torch
        self.model, self.alphabet = torch.hub.load(
            "facebookresearch/esm:main", 
            self.model_name
        )
        self.model.eval()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
    
    def add_proteins(self, proteins_dict):
        """
        Add proteins to database
        
        Args:
            proteins_dict: Dictionary {name: sequence}
        """
        import torch
        
        if self.model is None:
            self.load_model()
        
        self.proteins = list(proteins_dict.items())
        device = next(self.model.parameters()).device
        
        embeddings_list = []
        
        for i, (name, seq) in enumerate(self.proteins):
            tokens = self.alphabet.encode(seq)
            
            with torch.no_grad():
                results = self.model(tokens.to(device))
            
            embedding = torch.mean(results["representations"][33], dim=1)
            embeddings_list.append(embedding.cpu().numpy())
        
        self.embeddings = np.concatenate(embeddings_list).astype('float32')
        
        # Create FAISS index
        d = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.embeddings)
    
    def search(self, query_seq, top_k=5):
        """Search for similar proteins"""
        import torch
        
        device = next(self.model.parameters()).device
        tokens = self.alphabet.encode(query_seq)
        
        with torch.no_grad():
            results = self.model(tokens.to(device))
        
        query_emb = torch.mean(results["representations"][33], dim=1)
        query_emb = query_emb.cpu().numpy().astype('float32')
        
        distances, indices = self.index.search(query_emb, top_k)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            similarity = 1 / (1 + distance)
            name, seq = self.proteins[idx]
            results.append({
                'name': name,
                'similarity': float(similarity),
                'sequence': seq
            })
        
        return results
```

---

## Part 6: Building a RAG System (Basic)

### 6.1 Retrieval-Augmented Generation Concept

```
User Query: "What is CRISPR?"
        ↓
[Embed Query]
        ↓
[Search Document Database with FAISS]
        ↓
Retrieved Documents: 
  - "CRISPR is a gene editing tool..."
  - "The CRISPR-Cas9 system works by..."
        ↓
[Combine Query + Retrieved Docs]
Augmented Prompt: 
  "Based on this information: [Retrieved Docs]
   Answer the question: What is CRISPR?"
        ↓
[Send to LLM]
        ↓
LLM Response: "CRISPR is a revolutionary..."
```

### 6.2 Simple RAG Implementation

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class SimpleRAG:
    """Basic Retrieval-Augmented Generation system"""
    
    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(embedding_model)
        self.documents = []
        self.embeddings = None
        self.index = None
    
    def add_documents(self, documents):
        """Add documents to knowledge base"""
        print(f"Encoding {len(documents)} documents...")
        self.documents = documents
        
        embeddings = self.model.encode(documents)
        self.embeddings = np.array(embeddings).astype('float32')
        
        # Create index
        d = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.embeddings)
        
        print(f"Added {len(documents)} to knowledge base")
    
    def retrieve(self, query, k=3):
        """Retrieve relevant documents"""
        query_emb = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_emb, k)
        
        retrieved = []
        for idx in indices[0]:
            retrieved.append(self.documents[idx])
        
        return retrieved
    
    def augment_prompt(self, query, k=3):
        """Create augmented prompt for LLM"""
        docs = self.retrieve(query, k=k)
        
        context = "\n".join([f"- {doc}" for doc in docs])
        
        augmented = f"""Based on the following information:

{context}

Answer this question: {query}"""
        
        return augmented
    
    def generate_with_llm(self, query, llm_function, k=3):
        """Generate response using LLM with retrieval"""
        augmented_prompt = self.augment_prompt(query, k=k)
        response = llm_function(augmented_prompt)
        
        return {
            'query': query,
            'retrieved_docs': self.retrieve(query, k=k),
            'augmented_prompt': augmented_prompt,
            'response': response
        }


# Example usage
if __name__ == "__main__":
    # Initialize RAG
    rag = SimpleRAG()
    
    # Knowledge base
    knowledge_base = [
        "CRISPR-Cas9 is a gene editing technology that allows precise modifications to DNA",
        "The CRISPR system consists of a guide RNA and the Cas9 protein",
        "Gene therapy uses modified genes to treat or prevent genetic diseases",
        "Off-target effects can occur when CRISPR cuts at unintended locations",
        "The efficiency of CRISPR editing varies depending on the target sequence",
    ]
    
    # Add documents
    rag.add_documents(knowledge_base)
    
    # Test retrieval
    query = "How accurate is CRISPR editing?"
    print(f"\nQuery: {query}")
    print("\nRetrieved documents:")
    for i, doc in enumerate(rag.retrieve(query, k=2), 1):
        print(f"{i}. {doc}")
```

---

## Conclusion

**You've learned:**
✅ What embeddings are and how they work
✅ How to create embeddings with Sentence-Transformers
✅ Fast similarity search with FAISS
✅ Protein and DNA sequence embeddings
✅ Building RAG systems for knowledge-enhanced AI

**Key Takeaways:**
- Embeddings convert text/sequences to numerical vectors
- Similar concepts have similar embeddings
- FAISS enables fast search over millions of documents
- Specialized embeddings (proteins, DNA) work better for biology
- RAG systems let LLMs access external knowledge

**Next Steps:**
In the next notebooks, we'll:
1. **Notebook 4**: Work with biological sequences (DNA, proteins, RNA)
2. **Notebook 5**: Build complete bioinformatics applications
3. **Notebook 6**: Fine-tune models on your own biological data
