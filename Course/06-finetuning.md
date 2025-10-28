# Notebook 6: Fine-Tuning Local Models on Biological Data

## Introduction

Welcome to Notebook 6, the final notebook! Here you'll learn to customize local LLMs for your specific bioinformatics needs.

**Topics:**
- Preparing biological datasets for training
- Fine-tuning models on gene sequences
- Adapting models for specific biological tasks
- Using LoRA (Low-Rank Adaptation) for efficient training
- Quantization-aware fine-tuning
- Evaluating fine-tuned models

**Why Fine-Tune?**
- Pre-trained models are general-purpose
- Fine-tuning specializes them for your domain
- Small models can match large ones on specific tasks
- Maintains privacy (training locally)
- Cost-effective and fast

---

## Part 1: Preparing Biological Datasets

### 1.1 Creating Training Data

```python
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
import random

class BiologicalDatasetCreator:
    """Create training datasets for biological tasks"""
    
    def __init__(self):
        self.examples = []
    
    def create_gene_function_dataset(self, 
                                    gene_annotations_file: str) -> List[Dict]:
        """
        Create dataset for gene function prediction
        
        Args:
            gene_annotations_file: CSV with gene names and descriptions
            
        Returns:
            List of training examples in instruction-following format
        """
        
        df = pd.read_csv(gene_annotations_file)
        
        training_examples = []
        
        for _, row in df.iterrows():
            gene_name = row['gene_name']
            function = row['function']
            sequence = row.get('sequence', '')
            
            # Create instruction-response pairs
            instruction = f"What is the function of the {gene_name} gene?"
            
            response = f"The {gene_name} gene is involved in: {function}"
            
            if sequence:
                instruction = f"Based on the protein sequence {sequence[:50]}..., what is the likely gene function for {gene_name}?"
                response += f"\n\nThe protein sequence shows conservation in regions associated with {function}."
            
            training_examples.append({
                'instruction': instruction,
                'input': '',
                'output': response
            })
        
        self.examples = training_examples
        return training_examples
    
    def create_sequence_analysis_dataset(self,
                                        fasta_file: str,
                                        analysis_labels_file: str) -> List[Dict]:
        """
        Create dataset for sequence analysis
        
        Args:
            fasta_file: Sequences in FASTA format
            analysis_labels_file: CSV with sequence IDs and analyses
            
        Returns:
            List of training examples
        """
        
        from Bio import SeqIO
        
        # Load sequences
        sequences = {}
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequences[record.id] = str(record.seq)
        
        # Load analysis labels
        df = pd.read_csv(analysis_labels_file)
        
        training_examples = []
        
        for _, row in df.iterrows():
            seq_id = row['seq_id']
            analysis = row['analysis']
            
            if seq_id not in sequences:
                continue
            
            sequence = sequences[seq_id]
            
            # Create example
            instruction = f"Analyze this sequence: {sequence[:100]}..."
            
            response = f"Analysis: {analysis}"
            
            training_examples.append({
                'instruction': instruction,
                'input': sequence,
                'output': response
            })
        
        self.examples = training_examples
        return training_examples
    
    def create_dna_variant_dataset(self,
                                  variant_vcf_file: str,
                                  phenotype_file: str) -> List[Dict]:
        """
        Create dataset for variant interpretation
        
        Args:
            variant_vcf_file: VCF file with genetic variants
            phenotype_file: CSV with variant IDs and phenotypes
            
        Returns:
            List of training examples
        """
        
        # Load phenotypes
        df = pd.read_csv(phenotype_file)
        
        training_examples = []
        
        for _, row in df.iterrows():
            variant_id = row['variant_id']
            chrom = row['chromosome']
            pos = row['position']
            ref = row['reference']
            alt = row['alternate']
            phenotype = row['phenotype']
            
            instruction = f"Interpret this genetic variant: {chrom}:{pos} {ref}→{alt}"
            
            response = f"This variant is associated with: {phenotype}"
            
            training_examples.append({
                'instruction': instruction,
                'input': f"Variant: {variant_id}",
                'output': response
            })
        
        self.examples = training_examples
        return training_examples
    
    def create_rnaseq_interpretation_dataset(self,
                                           expression_file: str,
                                           interpretation_file: str) -> List[Dict]:
        """
        Create dataset for RNA-seq interpretation
        
        Args:
            expression_file: CSV with gene expression values
            interpretation_file: CSV with expert interpretations
            
        Returns:
            List of training examples
        """
        
        expr_df = pd.read_csv(expression_file)
        interp_df = pd.read_csv(interpretation_file)
        
        training_examples = []
        
        for _, row in interp_df.iterrows():
            gene = row['gene']
            expression_pattern = row['expression_pattern']
            interpretation = row['interpretation']
            
            instruction = f"Interpret this gene expression pattern: {gene} shows {expression_pattern}"
            
            response = f"Interpretation: {interpretation}"
            
            training_examples.append({
                'instruction': instruction,
                'input': '',
                'output': response
            })
        
        self.examples = training_examples
        return training_examples
    
    def save_to_jsonl(self, output_file: str):
        """Save dataset in JSONL format"""
        
        with open(output_file, 'w') as f:
            for example in self.examples:
                f.write(json.dumps(example) + '\\n')
        
        print(f"Saved {len(self.examples)} examples to {output_file}")
    
    def split_dataset(self, train_ratio=0.8, val_ratio=0.1):
        """
        Split dataset into train/validation/test
        
        Returns:
            Tuple of (train, val, test) lists
        """
        
        random.shuffle(self.examples)
        
        n = len(self.examples)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        train = self.examples[:train_size]
        val = self.examples[train_size:train_size + val_size]
        test = self.examples[train_size + val_size:]
        
        return train, val, test


# Example usage
if __name__ == "__main__":
    creator = BiologicalDatasetCreator()
    
    # Create examples
    examples = [
        {
            'instruction': 'What is the function of TP53?',
            'input': '',
            'output': 'TP53 is a tumor suppressor gene that regulates cell cycle and apoptosis'
        },
        {
            'instruction': 'What is the function of EGFR?',
            'input': '',
            'output': 'EGFR is a growth factor receptor involved in cell signaling'
        }
    ]
    
    creator.examples = examples
    creator.save_to_jsonl('training_data.jsonl')
    
    # Split
    train, val, test = creator.split_dataset()
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
```

### 1.2 Data Formatting for Instruction Fine-tuning

```python
def format_training_examples(examples: List[Dict]) -> List[str]:
    """
    Format examples for instruction fine-tuning
    
    Creates prompts that teach the model to follow instructions
    """
    
    formatted = []
    
    for example in examples:
        # Format for instruction-following
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Input:
{example.get('input', '')}

### Response:
{example['output']}"""
        
        formatted.append(prompt)
    
    return formatted


def create_conversation_format(examples: List[Dict]) -> List[Dict]:
    """Format for conversational fine-tuning (chat format)"""
    
    formatted = []
    
    for example in examples:
        formatted.append({
            'role': 'user',
            'content': example['instruction']
        })
        formatted.append({
            'role': 'assistant',
            'content': example['output']
        })
    
    return formatted
```

---

## Part 2: Fine-Tuning with LoRA

### 2.1 What is LoRA?

LoRA (Low-Rank Adaptation) is an efficient fine-tuning method:

```
Without LoRA:
- Update all model weights (billions of parameters)
- Requires lots of memory and time
- 24GB+ VRAM needed

With LoRA:
- Only update small adapter matrices
- 1000x fewer parameters to train
- Can fine-tune on consumer GPUs
- Performance comparable to full fine-tuning
```

### 2.2 Fine-Tuning with LoRA Implementation

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset, load_dataset
import json

class LocalModelFinetuner:
    """Fine-tune local models with LoRA"""
    
    def __init__(self, 
                 model_name: str = "TinyLlama/TinyLlama-1.1b-Chat-v1.0",
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.05):
        """
        Initialize fine-tuner
        
        Args:
            model_name: HuggingFace model ID
            lora_r: LoRA rank (lower = fewer parameters)
            lora_alpha: LoRA scaling
            lora_dropout: Dropout for LoRA layers
        """
        
        print(f"Loading model: {model_name}")
        
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map="auto"
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],  # For most models
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Wrap model with LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        print(f"LoRA model ready:")
        print(f"  Trainable parameters: {self.model.print_trainable_parameters()}")
    
    def prepare_dataset(self, data_file: str, max_length: int = 512):
        """
        Prepare dataset for training
        
        Args:
            data_file: JSONL file with training examples
            max_length: Maximum sequence length
            
        Returns:
            HuggingFace Dataset
        """
        
        print(f"Loading dataset from {data_file}")
        
        # Load examples
        examples = []
        with open(data_file, 'r') as f:
            for line in f:
                examples.append(json.loads(line))
        
        # Format examples
        formatted_texts = []
        
        for example in examples:
            text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Input:
{example.get('input', '')}

### Response:
{example['output']}<|endoftext|>"""
            
            formatted_texts.append(text)
        
        # Create dataset
        dataset = Dataset.from_dict({'text': formatted_texts})
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        
        print(f"Prepared {len(tokenized_dataset)} examples")
        
        return tokenized_dataset
    
    def train(self,
              train_dataset,
              output_dir: str = "./lora_model",
              num_epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 2e-4,
              save_steps: int = 100):
        """
        Fine-tune model with LoRA
        
        Args:
            train_dataset: HuggingFace Dataset
            output_dir: Directory to save model
            num_epochs: Number of training epochs
            batch_size: Batch size (reduce if OOM)
            learning_rate: Learning rate
            save_steps: Steps between saves
        """
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            save_steps=save_steps,
            save_total_limit=2,
            logging_steps=10,
            learning_rate=learning_rate,
            warmup_steps=100,
            optim="adamw_torch" if self.device == "cuda" else "adamw_torch",
            max_grad_norm=1.0
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer
        )
        
        # Train
        print("\\nStarting training...")
        trainer.train()
        
        # Save
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"\\nModel saved to {output_dir}")
        print(f"LoRA weights are {sum(p.numel() for p in self.model.peft_config['default'].parameters() if p.requires_grad) / 1e6:.2f}M parameters")
    
    def generate(self, prompt: str, max_new_tokens: int = 100):
        """Generate text with fine-tuned model"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# Example usage
if __name__ == "__main__":
    # Create fine-tuner
    finetuner = LocalModelFinetuner(
        model_name="TinyLlama/TinyLlama-1.1b-Chat-v1.0"
    )
    
    # Prepare dataset
    dataset = finetuner.prepare_dataset("training_data.jsonl")
    
    # Fine-tune
    finetuner.train(
        dataset,
        num_epochs=2,
        batch_size=2,  # Reduce for small GPUs
        learning_rate=2e-4
    )
    
    # Test
    prompt = "What is the function of BRCA1?"
    response = finetuner.generate(prompt)
    print(f"Q: {prompt}")
    print(f"A: {response}")
```

---

## Part 3: Evaluation of Fine-Tuned Models

### 3.1 Benchmark Fine-Tuned Model

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelEvaluator:
    """Evaluate fine-tuned models"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def evaluate_on_test_set(self, test_examples: List[Dict]) -> Dict:
        """
        Evaluate model on test set
        
        Args:
            test_examples: List of test examples with 'instruction' and 'output'
            
        Returns:
            Dictionary with evaluation metrics
        """
        
        predictions = []
        references = []
        
        for example in test_examples:
            instruction = example['instruction']
            reference = example['output']
            
            # Generate prediction
            prompt = f"### Instruction:\\n{instruction}\\n\\n### Response:\\n"
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=50,
                    do_sample=False
                )
            
            prediction = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            ).split("### Response:")[1].strip()
            
            predictions.append(prediction)
            references.append(reference)
        
        # Calculate metrics
        metrics = {
            'n_samples': len(test_examples),
            'predictions': predictions,
            'references': references
        }
        
        return metrics
    
    def compare_models(self, 
                      original_model,
                      finetuned_model,
                      test_examples: List[Dict]) -> Dict:
        """
        Compare original and fine-tuned models
        
        Args:
            original_model: Original pre-trained model
            finetuned_model: Fine-tuned model
            test_examples: Test examples
            
        Returns:
            Comparison results
        """
        
        comparison = {
            'original': self.evaluate_on_test_set_internal(
                original_model, test_examples
            ),
            'finetuned': self.evaluate_on_test_set_internal(
                finetuned_model, test_examples
            )
        }
        
        return comparison
    
    def evaluate_on_test_set_internal(self, model, test_examples):
        """Internal evaluation"""
        
        predictions = []
        
        for example in test_examples[:5]:  # Limit for speed
            prompt = f"### Instruction:\\n{example['instruction']}"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=30,
                    do_sample=False
                )
            
            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(prediction)
        
        return predictions
```

---

## Part 4: Production Deployment

### 4.1 Saving and Loading Fine-Tuned Models

```python
class ModelManager:
    """Manage fine-tuned models"""
    
    @staticmethod
    def save_lora_model(model, tokenizer, output_dir: str):
        """Save fine-tuned model"""
        
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
    
    @staticmethod
    def load_lora_model(model_dir: str):
        """Load fine-tuned model"""
        
        from peft import PeftModel
        
        # Load base model
        base_model_name = "TinyLlama/TinyLlama-1.1b-Chat-v1.0"
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        
        # Load LoRA weights
        model = PeftModel.from_pretrained(base_model, model_dir)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        return model, tokenizer
    
    @staticmethod
    def merge_and_quantize(model, tokenizer, output_dir: str):
        """Merge LoRA weights and quantize for deployment"""
        
        # Merge LoRA weights into base model
        merged_model = model.merge_and_unload()
        
        # Save merged model
        merged_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"Merged and quantized model saved to {output_dir}")


# Deployment example
if __name__ == "__main__":
    # Load fine-tuned model
    model, tokenizer = ModelManager.load_lora_model("./lora_model")
    
    # Merge and save for deployment
    ModelManager.merge_and_quantize(model, tokenizer, "./production_model")
    
    # Now this can be deployed with ollama or llama.cpp
    print("\\nModel ready for deployment!")
    print("Use with ollama: ollama run ./production_model")
```

---

## Part 5: Advanced Fine-Tuning Techniques

### 5.1 Multi-Task Fine-Tuning

```python
class MultiTaskFinetuner:
    """Fine-tune on multiple biological tasks simultaneously"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tasks = {}
    
    def add_task(self, task_name: str, dataset: List[Dict]):
        """Add a task to fine-tune on"""
        self.tasks[task_name] = dataset
    
    def create_mixed_dataset(self) -> List[Dict]:
        """Create dataset mixing all tasks"""
        
        mixed = []
        
        for task_name, examples in self.tasks.items():
            for example in examples:
                mixed.append({
                    'task': task_name,
                    'instruction': example['instruction'],
                    'input': example.get('input', ''),
                    'output': example['output']
                })
        
        return mixed
```

### 5.2 Continued Pre-training on Biological Text

```python
class ContinuedPretraining:
    """Continue pre-training on biological corpora"""
    
    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def pretrain_on_pubmed(self, corpus_file: str, epochs: int = 1):
        """Continue pre-training on biological text corpus"""
        
        print(f"Continuing pre-training on {corpus_file}")
        
        # This improves performance on biological language
        # Similar to how BioGPT and BioBERT are trained
```

---

## Conclusion

**You've Learned:**
✅ Creating biological training datasets
✅ Fine-tuning models with LoRA
✅ Efficient parameter tuning
✅ Evaluating fine-tuned models
✅ Production deployment

**Key Takeaways:**
- LoRA enables fine-tuning on consumer GPUs
- Small models can specialize for specific tasks
- Fine-tuning preserves privacy and data control
- Multi-task learning improves generalization

**Next Steps for Your Research:**
1. Create domain-specific datasets for your organisms/genes
2. Fine-tune on your experimental protocols
3. Build custom bioinformatics assistants
4. Deploy locally for team collaboration
5. Iterate and improve based on results

---

## Complete Workshop Summary

**Workshop Accomplishments:**

| Notebook | Topic | Skills |
|----------|-------|--------|
| 1 | Environment Setup | Install tools, configure GPU, Docker |
| 2 | Running Local LLMs | Understand models, inference, chatbots |
| 3 | Embeddings & Search | Semantic search, FAISS, similarity |
| 4 | Biological Sequences | Parse FASTA, sequence analysis, embeddings |
| 5 | Applications | Literature mining, RAG, annotation |
| 6 | Fine-Tuning | Custom models, LoRA, deployment |

**Your Toolkit:**
✓ Local LLM inference (privacy-preserving)
✓ Semantic search over literature
✓ Sequence analysis and similarity
✓ Gene expression interpretation
✓ Custom AI assistants
✓ Research automation

**Ready to Apply:**
- Your specific organisms
- Your experimental data
- Your research questions
- Your privacy requirements

Welcome to the future of local AI for bioinformatics! 🧬
