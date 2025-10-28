# Notebook 4: Working with Biological Sequences - DNA, RNA, and Proteins

## Introduction

Welcome to Notebook 4! This is where we apply everything you've learned to real bioinformatics tasks. You'll learn:

**Topics:**
- Parsing biological sequence files (FASTA, FASTQ, GenBank)
- Analyzing DNA, RNA, and protein sequences
- Using AI models for biological sequence analysis
- Building sequence search tools
- Predicting protein properties with local LLMs
- Creating bioinformatics pipelines

**Why This Matters:**
- Process genomic data without uploading to cloud services
- Analyze sequences locally with complete privacy
- Integrate AI with standard bioinformatics workflows
- Speed up research with intelligent tools

---

## Part 1: Sequence File Handling with Biopython

### 1.1 Understanding Sequence Formats

**FASTA Format:**
```
>sequence_id [description]
ACGTACGTACGTACGT
ACGTACGTACGTACGT

>Homo_sapiens_hemoglobin_alpha
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHG
```

**Key Components:**
- `>` = Header marker
- `sequence_id` = Unique identifier
- Everything after space = Description (optional)
- Following lines = Sequence data (no spaces)

**FASTQ Format (DNA/RNA with quality scores):**
```
@sequence_id [description]
ACGTACGTACGTACGT
+[optional repeat id]
IIIIIIIIIIIIIIII

# Quality scores use ASCII characters
# I (ASCII 73) = high quality (Q40)
# ! (ASCII 33) = low quality (Q0)
```

### 1.2 Reading Sequence Files

```python
from Bio import SeqIO
from pathlib import Path

# ===== Reading FASTA Files =====

# Method 1: Parse single FASTA
fasta_file = "sequences.fasta"

for record in SeqIO.parse(fasta_file, "fasta"):
    print(f"ID: {record.id}")
    print(f"Description: {record.description}")
    print(f"Sequence: {str(record.seq)[:50]}...")
    print(f"Length: {len(record.seq)}")
    print()

# Method 2: Load all sequences into memory
sequences = list(SeqIO.parse(fasta_file, "fasta"))
print(f"Total sequences: {len(sequences)}")

# Method 3: Index FASTA for random access
indexed = SeqIO.index(fasta_file, "fasta")
print(f"Sequence IDs: {list(indexed.keys())}")

# Access specific sequence
seq_record = indexed["sequence_id_1"]
print(f"Sequence 1: {seq_record.seq}")

# ===== Reading FASTQ Files (with quality) =====

fastq_file = "reads.fastq"

for record in SeqIO.parse(fastq_file, "fastq"):
    print(f"ID: {record.id}")
    print(f"Sequence: {str(record.seq)[:50]}")
    print(f"Quality scores: {record.letter_annotations['phred_quality'][:50]}")
    print()

# Filter by quality
min_quality = 20
good_reads = []

for record in SeqIO.parse(fastq_file, "fastq"):
    avg_quality = sum(record.letter_annotations['phred_quality']) / len(record.seq)
    if avg_quality >= min_quality:
        good_reads.append(record)

print(f"Reads with average quality >= {min_quality}: {len(good_reads)}")
```

### 1.3 Working with Sequence Objects

```python
from Bio.Seq import Seq
from Bio import Alphabet

# Create sequence objects
dna_seq = Seq("ACGTACGTACGTACGT")
rna_seq = Seq("ACGUACGUACGUACGU")
protein_seq = Seq("MVLSPADKTNVKAAWGKVGAHAGEYGAEAL")

# Properties
print(f"DNA sequence: {dna_seq}")
print(f"Length: {len(dna_seq)}")
print(f"GC content: {(dna_seq.count('G') + dna_seq.count('C')) / len(dna_seq) * 100:.1f}%")

# Transcription (DNA → RNA)
rna_from_dna = dna_seq.transcribe()
print(f"DNA: {dna_seq}")
print(f"RNA: {rna_from_dna}")

# Translation (RNA/DNA → Protein)
protein = dna_seq.translate()
print(f"Protein: {protein}")

# Reverse complement (important for genomics)
complement = dna_seq.reverse_complement()
print(f"Original: {dna_seq}")
print(f"Reverse complement: {complement}")

# Codon usage
codons = [str(dna_seq[i:i+3]) for i in range(0, len(dna_seq)-2, 3)]
from collections import Counter
print(f"Codon frequencies: {Counter(codons)}")

# Find patterns
from Bio import Restriction
enz = Restriction.BamHI  # Example: BamHI restriction enzyme
sites = enz.search(dna_seq)
print(f"BamHI sites at positions: {sites}")
```

---

## Part 2: Sequence Analysis with AI Models

### 2.1 DNA Sequence Embedding and Analysis

```python
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class DNASequenceAnalyzer:
    """Analyze DNA sequences using DNA-BERT"""
    
    def __init__(self, model_name="dnabert/dnabert-2-117m"):
        print(f"Loading {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def get_embedding(self, sequence):
        """
        Get embedding for a DNA sequence
        
        Args:
            sequence: DNA sequence string (uppercase ACGT only)
            
        Returns:
            Numpy array of embeddings
        """
        
        # Validate sequence
        sequence = sequence.upper()
        valid_chars = set('ACGT')
        if not all(c in valid_chars for c in sequence):
            raise ValueError(f"Invalid DNA characters. Must be ACGT only.")
        
        # Tokenize
        inputs = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        return embedding
    
    def analyze_gene_region(self, sequence, window_size=100):
        """
        Analyze gene regions with sliding window
        
        Args:
            sequence: Full DNA sequence
            window_size: Size of analysis window
            
        Returns:
            List of embeddings for each window
        """
        
        embeddings = []
        positions = []
        
        for i in range(0, len(sequence) - window_size, window_size // 2):
            window = sequence[i:i + window_size]
            emb = self.get_embedding(window)
            embeddings.append(emb)
            positions.append(i)
        
        return embeddings, positions
    
    def find_similar_sequences(self, target_seq, search_space, threshold=0.85):
        """
        Find similar sequences in search space
        
        Args:
            target_seq: Target sequence
            search_space: Sequence to search within
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of similar regions with positions
        """
        
        target_emb = self.get_embedding(target_seq)
        window_size = len(target_seq)
        
        from scipy.spatial.distance import cosine
        
        similar_regions = []
        
        for i in range(0, len(search_space) - window_size, window_size // 2):
            window = search_space[i:i + window_size]
            window_emb = self.get_embedding(window)
            
            similarity = 1 - cosine(target_emb, window_emb)
            
            if similarity >= threshold:
                similar_regions.append({
                    'position': i,
                    'sequence': window,
                    'similarity': float(similarity)
                })
        
        return similar_regions


# Example usage
if __name__ == "__main__":
    analyzer = DNASequenceAnalyzer()
    
    # Example sequences
    gene1 = "ACGTACGTACGTACGTACGTACGTACGTACGT"
    gene2 = "ACGTACGTACGTACGTACGTACGTACGTACGT"  # Same
    gene3 = "TGCATGCATGCATGCATGCATGCATGCATGCA"  # Different
    
    # Get embeddings
    emb1 = analyzer.get_embedding(gene1)
    emb2 = analyzer.get_embedding(gene2)
    emb3 = analyzer.get_embedding(gene3)
    
    # Calculate similarity
    from scipy.spatial.distance import cosine
    
    sim_1_2 = 1 - cosine(emb1, emb2)
    sim_1_3 = 1 - cosine(emb1, emb3)
    
    print(f"Similarity (gene1 - gene2): {sim_1_2:.4f}")  # ~1.0
    print(f"Similarity (gene1 - gene3): {sim_1_3:.4f}")  # Lower
```

### 2.2 Protein Sequence Analysis

```python
class ProteinSequenceAnalyzer:
    """Analyze protein sequences using ESM2"""
    
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D"):
        print(f"Loading {model_name}...")
        
        self.model_name = model_name
        self.model, self.alphabet = torch.hub.load(
            "facebookresearch/esm:main",
            model_name
        )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def get_embedding(self, sequence):
        """Get protein sequence embedding"""
        
        tokens = self.alphabet.encode(sequence)
        
        with torch.no_grad():
            results = self.model(tokens.to(self.device))
        
        # Last layer, mean pooling
        embedding = torch.mean(
            results["representations"][33],  # Last layer
            dim=1
        ).cpu().numpy()
        
        return embedding
    
    def get_token_embeddings(self, sequence):
        """Get per-token embeddings (for position-specific analysis)"""
        
        tokens = self.alphabet.encode(sequence)
        
        with torch.no_grad():
            results = self.model(tokens.to(self.device))
        
        # Per-token representations
        token_embeddings = results["representations"][33].cpu().numpy()
        
        return token_embeddings, sequence
    
    def calculate_conservation(self, sequence_alignment):
        """
        Calculate sequence conservation from alignment
        
        Args:
            sequence_alignment: List of aligned protein sequences
        """
        
        embeddings = [self.get_embedding(seq) for seq in sequence_alignment]
        embeddings = np.array(embeddings)
        
        # Variance as measure of conservation (low = conserved)
        conservation = np.var(embeddings, axis=0)
        
        return conservation
    
    def predict_function(self, sequence):
        """
        Predict protein function from sequence
        (Using LLM for functional annotation)
        """
        
        prompt = f"""Based on the protein sequence below, predict its likely function and characteristics:

Sequence: {sequence}

Predicted function:"""
        
        # This would use an LLM to generate predictions
        # (See Notebook 2 for LLM integration)
        return prompt  # Return prompt for LLM


# Example usage
if __name__ == "__main__":
    analyzer = ProteinSequenceAnalyzer()
    
    # Example proteins
    hemoglobin = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHG"
    insulin = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT"
    
    # Get embeddings
    hemo_emb = analyzer.get_embedding(hemoglobin)
    insulin_emb = analyzer.get_embedding(insulin)
    
    # Similarity
    from scipy.spatial.distance import cosine
    sim = 1 - cosine(hemo_emb, insulin_emb)
    
    print(f"Hemoglobin-Insulin similarity: {sim:.4f}")
    
    # Per-token embeddings (for identifying functional domains)
    token_embs, seq = analyzer.get_token_embeddings(hemoglobin[:20])
    print(f"Token embeddings shape: {token_embs.shape}")
    print(f"(20 amino acids, 320-dimensional embeddings)")
```

---

## Part 3: Building a Genomics Search Tool

### 3.1 Complete Genome Search Pipeline

```python
import faiss
import numpy as np
from Bio import SeqIO

class GenomeSearchEngine:
    """Search genomic databases for similar regions"""
    
    def __init__(self, analyzer):
        """
        Initialize genome search engine
        
        Args:
            analyzer: DNASequenceAnalyzer instance
        """
        self.analyzer = analyzer
        self.sequences = []
        self.names = []
        self.embeddings = None
        self.index = None
    
    def index_genome(self, fasta_file, window_size=100):
        """
        Index a genome file for search
        
        Args:
            fasta_file: Path to FASTA file
            window_size: Size of indexing windows
        """
        
        print(f"Indexing {fasta_file}...")
        
        all_embeddings = []
        sequence_info = []
        
        # Process each sequence in FASTA
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq_str = str(record.seq)
            
            # Split into windows
            for i in range(0, len(seq_str) - window_size, window_size // 2):
                window = seq_str[i:i + window_size]
                
                try:
                    emb = self.analyzer.get_embedding(window)
                    all_embeddings.append(emb)
                    
                    sequence_info.append({
                        'sequence_id': record.id,
                        'position': i,
                        'sequence': window
                    })
                except:
                    continue  # Skip invalid sequences
        
        # Build FAISS index
        embeddings_array = np.vstack(all_embeddings).astype('float32')
        
        d = embeddings_array.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings_array)
        
        self.embeddings = embeddings_array
        self.sequence_info = sequence_info
        
        print(f"Indexed {len(sequence_info)} genomic regions")
    
    def search(self, query_sequence, top_k=5):
        """
        Search for similar genomic regions
        
        Args:
            query_sequence: Query DNA sequence
            top_k: Number of results
            
        Returns:
            List of similar regions
        """
        
        query_emb = self.analyzer.get_embedding(query_sequence)
        query_emb = query_emb.astype('float32')
        
        distances, indices = self.index.search(query_emb, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity = 1 / (1 + dist)
            info = self.sequence_info[idx]
            info['similarity'] = float(similarity)
            results.append(info)
        
        return results


# Example: Find similar genes
if __name__ == "__main__":
    analyzer = DNASequenceAnalyzer()
    search_engine = GenomeSearchEngine(analyzer)
    
    # Index a genome (example with synthetic data)
    import tempfile
    from Bio.SeqRecord import SeqRecord
    
    # Create example sequences
    example_seqs = [
        SeqRecord(Seq("ACGTACGTACGTACGT"), id="gene1"),
        SeqRecord(Seq("TGCATGCATGCATGCA"), id="gene2"),
    ]
    
    # Save to temporary FASTA
    temp_fasta = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)
    SeqIO.write(example_seqs, temp_fasta.name, "fasta")
    temp_fasta.close()
    
    # Index and search
    search_engine.index_genome(temp_fasta.name, window_size=8)
    
    # Search
    query = "ACGTACGTACGTACGT"
    results = search_engine.search(query, top_k=2)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Gene {result['sequence_id']}: similarity {result['similarity']:.4f}")
```

---

## Part 4: RNA-seq Analysis Integration

### 4.1 Processing Gene Expression Data

```python
import pandas as pd
import numpy as np
from pathlib import Path

class GeneExpressionAnalyzer:
    """Analyze gene expression data with AI"""
    
    def __init__(self, llm_model=None):
        self.llm_model = llm_model  # Optional LLM for interpretation
        self.expression_data = None
        self.genes = []
        self.samples = []
    
    def load_expression_matrix(self, filename):
        """
        Load expression matrix
        
        Args:
            filename: CSV/TSV with genes as rows, samples as columns
        """
        
        # Detect delimiter
        if filename.endswith('.csv'):
            sep = ','
        else:
            sep = '\\t'
        
        df = pd.read_csv(filename, sep=sep, index_col=0)
        
        self.expression_data = df
        self.genes = df.index.tolist()
        self.samples = df.columns.tolist()
        
        print(f"Loaded {len(self.genes)} genes across {len(self.samples)} samples")
        return df
    
    def normalize_expression(self, method='log2'):
        """
        Normalize expression data
        
        Args:
            method: 'log2', 'zscore', or 'quantile'
        """
        
        df = self.expression_data.copy()
        
        if method == 'log2':
            df = np.log2(df + 1)  # Add 1 to avoid log(0)
        
        elif method == 'zscore':
            from scipy import stats
            df = df.apply(stats.zscore, axis=1)
        
        elif method == 'quantile':
            # Quantile normalization
            from sklearn.preprocessing import QuantileTransformer
            qt = QuantileTransformer(output_distribution='normal')
            df = pd.DataFrame(
                qt.fit_transform(df.T).T,
                index=df.index,
                columns=df.columns
            )
        
        self.expression_data = df
        return df
    
    def find_differentially_expressed_genes(self, group1, group2, threshold=2):
        """
        Find differentially expressed genes between groups
        
        Args:
            group1: List of sample names for group 1
            group2: List of sample names for group 2
            threshold: Log2 fold-change threshold
        """
        
        # Calculate mean expression for each group
        mean_g1 = self.expression_data[group1].mean(axis=1)
        mean_g2 = self.expression_data[group2].mean(axis=1)
        
        # Calculate fold change
        fc = np.log2(mean_g2 / (mean_g1 + 1) + 1)
        
        # Find significant changes
        up_regulated = self.genes[fc > threshold]
        down_regulated = self.genes[fc < -threshold]
        
        return {
            'up_regulated': up_regulated.tolist(),
            'down_regulated': down_regulated.tolist(),
            'fold_changes': fc
        }
    
    def interpret_with_llm(self, genes_list, prompt_template=None):
        """
        Use LLM to interpret gene lists
        
        Args:
            genes_list: List of gene names
            prompt_template: Custom prompt template
        """
        
        if self.llm_model is None:
            return "LLM not configured"
        
        if prompt_template is None:
            prompt_template = """Based on the following list of genes: {genes}

Describe:
1. What biological processes these genes are involved in
2. What diseases or conditions might be affected
3. Potential therapeutic targets"""
        
        genes_str = ", ".join(genes_list[:20])  # Limit to 20 genes
        
        prompt = prompt_template.format(genes=genes_str)
        
        # This would call the LLM
        # (See Notebook 2 for implementation)
        
        return prompt


# Example usage
if __name__ == "__main__":
    # Create sample expression data
    np.random.seed(42)
    n_genes = 20000
    n_samples = 10
    
    expression_matrix = np.random.poisson(10, (n_genes, n_samples))
    
    df = pd.DataFrame(
        expression_matrix,
        index=[f"GENE_{i}" for i in range(n_genes)],
        columns=[f"sample_{i}" for i in range(n_samples)]
    )
    
    # Save temporarily
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name)
    temp_file.close()
    
    # Analyze
    analyzer = GeneExpressionAnalyzer()
    analyzer.load_expression_matrix(temp_file.name)
    analyzer.normalize_expression('log2')
    
    # Find DE genes
    group1 = [f"sample_{i}" for i in range(5)]
    group2 = [f"sample_{i}" for i in range(5, 10)]
    
    de_results = analyzer.find_differentially_expressed_genes(group1, group2)
    
    print(f"Up-regulated genes: {len(de_results['up_regulated'])}")
    print(f"Down-regulated genes: {len(de_results['down_regulated'])}")
```

---

## Part 5: Privacy-Preserving Bioinformatics Pipeline

### 5.1 Complete Local Pipeline

```python
class LocalBioinfoPipeline:
    """End-to-end local bioinformatics pipeline"""
    
    def __init__(self):
        self.dna_analyzer = None
        self.protein_analyzer = None
        self.expression_analyzer = None
        self.search_engine = None
    
    def initialize_all_components(self):
        """Initialize all analysis components"""
        
        print("Initializing DNA analyzer...")
        self.dna_analyzer = DNASequenceAnalyzer()
        
        print("Initializing protein analyzer...")
        self.protein_analyzer = ProteinSequenceAnalyzer()
        
        self.expression_analyzer = GeneExpressionAnalyzer()
        
        self.search_engine = GenomeSearchEngine(self.dna_analyzer)
    
    def analyze_sequencing_results(self, fastq_file):
        """Analyze sequencing data"""
        
        print(f"Processing {fastq_file}...")
        
        quality_scores = []
        sequence_count = 0
        
        for record in SeqIO.parse(fastq_file, "fastq"):
            sequence_count += 1
            avg_quality = sum(record.letter_annotations['phred_quality']) / len(record.seq)
            quality_scores.append(avg_quality)
        
        return {
            'total_reads': sequence_count,
            'avg_quality': np.mean(quality_scores),
            'min_quality': np.min(quality_scores),
            'max_quality': np.max(quality_scores)
        }
    
    def process_all_data(self, genome_file, transcriptome_file, expression_file):
        """Process all biological data locally"""
        
        results = {
            'sequencing_qc': self.analyze_sequencing_results(transcriptome_file),
            'genome_indexed': False,
            'expression_analyzed': False
        }
        
        # Index genome
        print("\\nIndexing genome...")
        self.search_engine.index_genome(genome_file)
        results['genome_indexed'] = True
        
        # Analyze expression
        print("\\nAnalyzing expression...")
        self.expression_analyzer.load_expression_matrix(expression_file)
        self.expression_analyzer.normalize_expression('log2')
        results['expression_analyzed'] = True
        
        return results


# Usage
if __name__ == "__main__":
    pipeline = LocalBioinfoPipeline()
    
    # All processing happens locally
    # No data leaves your computer
    # HIPAA/GDPR compliant
    
    print("✓ Bioinformatics pipeline ready")
    print("✓ All processing local and private")
```

---

## Conclusion

**Key Skills Learned:**
✅ Parse biological sequence files (FASTA, FASTQ)
✅ Analyze DNA, RNA, and protein sequences
✅ Use embeddings for sequence similarity
✅ Build genomic search tools
✅ Integrate with gene expression analysis
✅ Create privacy-preserving pipelines

**Next Notebook:**
In Notebook 5, we'll combine all these skills into complete bioinformatics applications that integrate LLMs with sequence analysis!
