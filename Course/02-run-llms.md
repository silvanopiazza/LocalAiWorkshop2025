# Notebook 2: Running Local LLMs - Understanding Models and Inference

## Introduction

Welcome to Notebook 2! In this notebook, you will learn:
- What Large Language Models (LLMs) are and how they work
- Different types of LLM architectures
- How to download and run models locally
- How to use different frameworks (Ollama, llama.cpp, Hugging Face)
- How to optimize inference for speed and memory
- How to create a simple chat application with local models

By the end, you'll have hands-on experience running multiple LLMs locally and understanding their strengths/weaknesses.

---

## Part 1: Understanding Large Language Models (LLMs)

### 1.1 What is a Large Language Model?

A Large Language Model is a neural network trained on enormous amounts of text data to predict the next word in a sequence. Think of it as an extremely sophisticated autocomplete system.

**Key Characteristics:**
- **Large**: Billions to trillions of parameters (weights)
- **Language**: Trained on text to understand and generate human language
- **Model**: A mathematical function mapping input text to output text

**How LLMs Work:**

```
Input: "DNA sequencing is a technique to"
                    ↓
[Tokenization] → [Neural Network Processing] → [Output Predictions]
                                                 ↓
Output: "determine" (with 95% confidence)
        "identify" (with 3% confidence)
        "measure" (with 2% confidence)
```

The model picks "determine" (highest probability), then repeats the process with the new input "DNA sequencing is a technique to determine"

### 1.2 Transformer Architecture

All modern LLMs use the **Transformer** architecture, introduced in 2017. It uses:

**Attention Mechanism**: Allows the model to focus on relevant parts of the input when generating each output token

```
Example: Processing "The patient's RNA was extracted from the blood sample"

When predicting the next word after "extracted from the", 
the model pays attention to:
- "RNA" (high attention - what was extracted)
- "patient's" (moderate attention - whose RNA)
- "blood sample" (high attention - where from)
- Other words (low attention)
```

**Key Components:**

1. **Tokenizer**: Splits text into small pieces (tokens)
   - Example: "Bioinformatics" → ["Bio", "infor", "matics"]
   - Each token gets a numerical ID

2. **Embedding Layer**: Converts token IDs to numerical vectors (embeddings)
   - Each token becomes a vector of 768-4096 numbers
   - Similar words have similar vectors

3. **Attention Layers**: Process embeddings, letting tokens attend to other tokens
   - Multiple layers (12 to 80+ depending on model size)
   - Each layer refines understanding

4. **Output Layer**: Predicts probabilities for all possible next tokens

### 1.3 Model Sizes and Parameters

**What are Parameters?**
Parameters are the weights in the neural network. Each parameter is a number learned during training.

**Common LLM Sizes:**

| Model Size | Parameters | Best For | Memory (VRAM) | Speed (Tokens/sec) |
|------------|-----------|----------|---------------|-------------------|
| Tiny | 1B | Local machines, demos | 2-4GB | 50-100 |
| Small | 7B | Laptops, edge devices | 4-8GB | 20-50 |
| Medium | 13B | Workstations | 8-16GB | 10-20 |
| Large | 34-70B | High-end GPUs | 40-80GB | 5-15 |
| Very Large | 100B+ | Enterprise servers | 100GB+ | 1-5 |

**Why Bigger Isn't Always Better:**
- Larger models are more capable but slower
- Trade-off between quality and speed
- For specific tasks, 7B can match 70B performance with fine-tuning

### 1.4 Quantization - The Secret to Local Inference

**What is Quantization?**
Quantization reduces the precision of model weights, dramatically reducing memory and speeding up inference.

**Precision Levels:**

| Precision | Bits | Size | Accuracy Loss | Example |
|-----------|------|------|---------------|---------|
| Full (FP32) | 32 | 100% | None | 0.247568 |
| Half (FP16) | 16 | 50% | <1% | 0.248 |
| 8-bit (INT8) | 8 | 25% | ~1-2% | 0 or 1 |
| 4-bit (INT4) | 4 | 12.5% | ~2-4% | 0, 0.25, 0.5, ... |

**Practical Example:**
```
Full Precision (FP32): 3.141592653589793
Half Precision (FP16): 3.1406
8-bit Quantized: 3 or 4
4-bit Quantized: 3
```

**Memory Savings:**
- 7B parameter model:
  - FP32: 7B × 4 bytes = 28GB
  - FP16: 7B × 2 bytes = 14GB
  - INT8: 7B × 1 byte = 7GB
  - INT4: 7B × 0.5 bytes = 3.5GB

**GGUF Format:**
GGUF (GPT-Generated Unified Format) is a format specifically designed for quantized models:
- Supports multiple quantization levels in one file
- Optimized for inference speed
- Used by llama.cpp and Ollama

---

## Part 2: Running Models with Different Frameworks

### 2.1 Using Ollama - The Easiest Way

**Why Ollama?**
- Simplest interface: `ollama run model_name`
- Automatic model management
- Built-in REST API
- Works on Mac, Linux, Windows

**Starting Ollama Server:**

```bash
# Start the Ollama service
ollama serve

# In another terminal, verify it's running
curl http://localhost:11434/api/tags
```

**Running Your First Model:**

```bash
# Simple chat
ollama run llama2

# With a specific prompt
ollama run llama2 "What is the central dogma of molecular biology?"

# With parameters
ollama run llama2 --temperature 0.7 --num-predict 256 \
  "Explain CRISPR gene editing in simple terms"
```

**Understanding Parameters:**

- **temperature**: Controls randomness
  - 0.0 = Deterministic (always same answer)
  - 0.5 = Balanced (creative but coherent)
  - 1.0+ = Very creative (can be nonsensical)

- **num-predict**: Maximum tokens to generate
  - Default: 128
  - Higher = longer responses

- **top-p**: Nucleus sampling (diversity)
  - 0.9 = Consider top 90% of probable tokens

**Available Biology-Relevant Models:**

```bash
ollama pull mistral          # Fast, good for quick tasks
ollama pull neural-chat      # Optimized for conversation
ollama pull orca-mini        # Small, instruction-tuned
ollama pull openchat         # General purpose
```

**Check Available Models:**

```bash
ollama list

# Output:
# NAME              ID              SIZE    MODIFIED
# llama2:latest     46e6a4676fef    3.8 GB  2 hours ago
# mistral:latest    61e88e884507    4.1 GB  1 day ago
```

### 2.2 Using llama.cpp - Fine-Grained Control

**Why llama.cpp?**
- Highly optimized C++ implementation
- CPU inference option (no GPU required)
- Most control over parameters
- Fastest inference

**Downloading Models:**

GGUF models are hosted on Hugging Face. Here's how to find them:

```bash
# Create directory for models
mkdir -p ~/llm_models
cd ~/llm_models

# Download a small model (TinyLlama - 1.1B parameters)
# This is good for testing on limited hardware
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Download Mistral 7B (more capable, still small)
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/Mistral-7B-Instruct-v0.1.Q4_K_M.gguf

# Download Llama 2 7B
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
```

**Understanding Model Names:**

- **TinyLlama-1.1B**: 1.1 billion parameters
- **Mistral-7B**: 7 billion parameters
- **Q4_K_M**: Quantization level
  - Q4 = 4-bit quantization
  - K = Special algorithm
  - M = Medium balance between speed/quality

**Running with llama.cpp:**

```bash
# Navigate to llama.cpp
cd ~/llama.cpp

# Simple inference
./main -m ~/llm_models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -p "Explain DNA replication" \
  -n 256

# With more options
./main -m ~/llm_models/mistral-7b-instruct-v0.1.Q4_K_M.gguf \
  -p "What are the benefits of local AI for bioinformatics?" \
  -n 512 \
  --temperature 0.7 \
  --top-p 0.9 \
  -ngl 33  # GPU layers (33 = all layers on GPU if supported)
```

**Key Parameters:**

- `-m`: Model file path
- `-p`: Prompt text
- `-n`: Number of tokens to generate
- `--temperature`: Randomness (0-2)
- `--top-p`: Nucleus sampling
- `-ngl`: Number of GPU layers (0 = CPU only)
- `-t`: Number of threads to use
- `--ctx-size`: Context window size (max tokens to remember)

### 2.3 Using Hugging Face Transformers - Full Python Control

**Why Transformers?**
- Full Python API
- Direct access to model internals
- Easy integration in scripts
- Most flexible for research

**Basic Usage:**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Define model
model_name = "TinyLlama/TinyLlama-1.1b-Chat-v1.0"

# Load tokenizer (converts text to numbers)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model (the actual neural network)
print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"  # Automatically place on GPU/CPU
)

# Create a prompt
prompt = "What is CRISPR?"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate output
output_ids = model.generate(
    inputs["input_ids"],
    max_length=200,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

# Decode output to text
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Response: {response}")
```

**Understanding the Code:**

1. **AutoTokenizer.from_pretrained()**: Downloads and loads the tokenizer
   - Tokenizer converts words → numbers
   - Example: "DNA" → [4856, 29871]

2. **AutoModelForCausalLM.from_pretrained()**: Loads the model
   - "Causal" = predicts next token based on previous tokens
   - `torch_dtype=torch.float16`: Use half precision for speed/memory
   - `device_map="auto"`: Automatically split across GPU/CPU

3. **model.generate()**: Generates new tokens
   - `max_length=200`: Max output length
   - `temperature=0.7`: Some randomness
   - `do_sample=True`: Use sampling instead of greedy decoding

4. **tokenizer.decode()**: Converts numbers back to words

**Advanced Example with Chat Template:**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

# Create conversation
messages = [
    {"role": "user", "content": "What is the genetic code?"}
]

# Apply chat template
formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Tokenize and generate
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    inputs["input_ids"],
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id
)

# Decode
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

---

## Part 3: Building a Simple Chatbot

Let's create a reusable chatbot class that works with multiple backends.

### 3.1 Ollama-Based Chatbot

```python
import requests
import json

class OllamaChatbot:
    """Simple chatbot interface for Ollama models"""
    
    def __init__(self, model_name="mistral", api_url="http://localhost:11434"):
        """
        Initialize Ollama chatbot
        
        Args:
            model_name: Name of model to use (e.g., 'mistral', 'llama2')
            api_url: URL of Ollama API server
        """
        self.model_name = model_name
        self.api_url = api_url
        self.conversation_history = []
        
    def chat(self, user_message, temperature=0.7, top_p=0.9):
        """
        Send a message and get a response
        
        Args:
            user_message: User's question/statement
            temperature: Randomness (0.0-1.0)
            top_p: Nucleus sampling parameter
            
        Returns:
            Model's response as string
        """
        
        # Add to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Create prompt from history
        prompt = self._format_prompt()
        
        # Call Ollama API
        response = requests.post(
            f"{self.api_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature,
                "top_p": top_p
            }
        )
        
        if response.status_code != 200:
            return f"Error: {response.text}"
        
        # Extract response
        result = response.json()
        assistant_message = result["response"].strip()
        
        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def _format_prompt(self):
        """Format conversation history into a prompt"""
        prompt_parts = []
        for msg in self.conversation_history:
            if msg["role"] == "user":
                prompt_parts.append(f"User: {msg['content']}")
            else:
                prompt_parts.append(f"Assistant: {msg['content']}")
        return "\n".join(prompt_parts) + "\nAssistant:"
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


# Usage Example
if __name__ == "__main__":
    # Create chatbot
    bot = OllamaChatbot(model_name="mistral")
    
    # Multi-turn conversation
    print("Biology Chatbot (type 'quit' to exit)")
    print("=" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
            
        if not user_input:
            continue
        
        print("\nAssistant: ", end="", flush=True)
        response = bot.chat(user_input)
        print(response)
```

### 3.2 Hugging Face Transformers-Based Chatbot

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LocalLLMChatbot:
    """Chatbot using Hugging Face Transformers"""
    
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1b-Chat-v1.0"):
        """
        Initialize with a Hugging Face model
        
        Args:
            model_name: Model identifier from Hugging Face
        """
        print(f"Loading model: {model_name}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map="auto"
        )
        
        self.model.eval()  # Set to evaluation mode
        self.conversation_history = []
        
        print(f"Model loaded on {self.device}")
    
    def chat(self, user_message, max_length=256, temperature=0.7, top_p=0.9):
        """
        Generate a response to user message
        
        Args:
            user_message: User's input
            max_length: Maximum response length
            temperature: Randomness level
            top_p: Nucleus sampling parameter
            
        Returns:
            Model's response
        """
        
        # Add to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Format prompt
        prompt = self._format_prompt()
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate with torch.no_grad() to save memory
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        full_response = self.tokenizer.decode(
            output_ids[0], 
            skip_special_tokens=True
        )
        
        # Extract only the new part (remove prompt)
        response = full_response[len(prompt):].strip()
        
        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def _format_prompt(self):
        """Format conversation history"""
        prompt = ""
        for msg in self.conversation_history:
            if msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            else:
                prompt += f"Assistant: {msg['content']}\n"
        prompt += "Assistant:"
        return prompt
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


# Usage Example
if __name__ == "__main__":
    # For small models use TinyLlama
    # For better quality but slower, use Mistral
    bot = LocalLLMChatbot("TinyLlama/TinyLlama-1.1b-Chat-v1.0")
    
    print("\nLocal LLM Chatbot")
    print("=" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if not user_input:
            continue
        
        print("\nAssistant: ", end="", flush=True)
        response = bot.chat(user_input, max_length=256)
        print(response)
```

---

## Part 4: Comparing Model Architectures

### 4.1 Popular Open-Source Models for Biology

| Model | Size | Quantized Size | Best For | Speed | Quality |
|-------|------|----------------|----------|-------|---------|
| TinyLlama | 1.1B | ~600MB | Testing, demos | Very Fast | Basic |
| Mistral 7B | 7B | ~4GB | General tasks | Fast | Good |
| Llama 2 7B | 7B | ~4GB | General tasks | Fast | Good |
| OpenChat | 3.5B | ~2GB | Quick inference | Very Fast | Fair |
| Orca Mini 7B | 7B | ~4GB | Reasoning | Fast | Very Good |
| Neural Chat 7B | 7B | ~4GB | Conversation | Fast | Good |

### 4.2 Model Download Helper

```python
import os
import requests
from pathlib import Path

class ModelDownloader:
    """Download GGUF models from Hugging Face"""
    
    # Popular models for local inference
    POPULAR_MODELS = {
        "tinyllama": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "mistral": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/Mistral-7B-Instruct-v0.1.Q4_K_M.gguf",
        "llama2": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
        "openchat": "https://huggingface.co/TheBloke/Openchat-3.5-GGUF/resolve/main/openchat-3.5.Q4_K_M.gguf",
    }
    
    def __init__(self, download_dir="./models"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
    
    def download_model(self, model_key, show_progress=True):
        """
        Download a model
        
        Args:
            model_key: Key from POPULAR_MODELS (e.g., 'mistral')
            show_progress: Show download progress
            
        Returns:
            Path to downloaded model
        """
        
        if model_key not in self.POPULAR_MODELS:
            print(f"Unknown model: {model_key}")
            print(f"Available: {list(self.POPULAR_MODELS.keys())}")
            return None
        
        url = self.POPULAR_MODELS[model_key]
        filename = self.download_dir / url.split('/')[-1]
        
        if filename.exists():
            print(f"Model already exists: {filename}")
            return filename
        
        print(f"Downloading {model_key} from {url}")
        response = requests.get(url, stream=True)
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if show_progress and total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"Progress: {percent:.1f}%", end='\r')
        
        print(f"\nDownloaded to: {filename}")
        return filename


# Usage
if __name__ == "__main__":
    downloader = ModelDownloader("./models")
    
    # Download a model
    model_path = downloader.download_model("mistral")
    
    # List available
    print("Available models:", list(downloader.POPULAR_MODELS.keys()))
```

---

## Part 5: Measuring Performance

### 5.1 Benchmarking Script

```python
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class PerformanceBenchmark:
    """Benchmark model inference speed and memory usage"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map="auto"
        )
    
    def measure_speed(self, prompt, num_tokens=100, num_runs=3):
        """
        Measure generation speed
        
        Args:
            prompt: Input text
            num_tokens: Tokens to generate
            num_runs: Number of runs to average
        """
        
        times = []
        
        for run in range(num_runs):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            torch.cuda.synchronize() if self.device == "cuda" else None
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=num_tokens,
                    do_sample=False
                )
            
            torch.cuda.synchronize() if self.device == "cuda" else None
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        tokens_per_sec = num_tokens / avg_time
        
        return {
            "avg_time": avg_time,
            "tokens_per_second": tokens_per_sec,
            "times": times
        }
    
    def measure_memory(self):
        """Measure GPU/CPU memory usage"""
        
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "device": "CUDA"
            }
        else:
            import psutil
            process = psutil.Process()
            return {
                "memory_mb": process.memory_info().rss / 1e6,
                "device": "CPU"
            }


# Usage
if __name__ == "__main__":
    benchmark = PerformanceBenchmark("TinyLlama/TinyLlama-1.1b-Chat-v1.0")
    
    # Test prompt
    prompt = "What is the central dogma of molecular biology? "
    
    # Speed benchmark
    print("\nBenchmarking generation speed...")
    speed_results = benchmark.measure_speed(prompt, num_tokens=50, num_runs=3)
    print(f"Average time: {speed_results['avg_time']:.2f}s")
    print(f"Tokens/second: {speed_results['tokens_per_second']:.1f}")
    
    # Memory benchmark
    print("\nMemory usage:")
    mem_results = benchmark.measure_memory()
    for key, value in mem_results.items():
        print(f"  {key}: {value}")
```

---

## Part 6: Practical Examples for Biology

### 6.1 PubMed Abstract Summarizer

```python
class AbstractSummarizer:
    """Summarize PubMed abstracts using local LLM"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def summarize(self, abstract, max_length=100):
        """
        Summarize a scientific abstract
        
        Args:
            abstract: Text of the abstract
            max_length: Maximum summary length
        """
        
        prompt = f"""Summarize this scientific abstract in 2-3 key points:

Abstract: {abstract}

Summary:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_length,
                temperature=0.5,
                do_sample=False
            )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.replace(prompt, "").strip()


# Example usage
abstract = """
Background: CRISPR-Cas9 gene editing has revolutionized genetic engineering.
Methods: We developed a novel delivery system for CRISPR components.
Results: The system achieved 85% editing efficiency in primary cells.
Conclusion: This approach enables safer and more efficient gene therapy.
"""

# summarizer = AbstractSummarizer(model, tokenizer, device)
# summary = summarizer.summarize(abstract)
# print(summary)
```

---

## Part 7: Troubleshooting Common Issues

### 7.1 Out of Memory Errors

**Problem**: `torch.cuda.OutOfMemoryError: CUDA out of memory`

**Solutions:**

```python
# Solution 1: Use smaller model
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1b")

# Solution 2: Use lower precision
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16  # or torch.bfloat16
)

# Solution 3: Clear cache before generation
torch.cuda.empty_cache()
outputs = model.generate(...)

# Solution 4: Use load_in_4bit or load_in_8bit
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Solution 5: Reduce max_length
outputs = model.generate(..., max_new_tokens=100)  # Instead of 512
```

### 7.2 Slow Inference

**Problem**: Model generates very slowly (< 5 tokens/second)

**Diagnosis:**

```python
# Check if GPU is being used
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.current_device())  # Should be 0 or higher

# Check GPU utilization
import subprocess
subprocess.run(['nvidia-smi'])  # Look for GPU usage

# Check if model is on GPU
print(next(model.parameters()).device)  # Should show cuda:0, not cpu
```

**Solutions:**

```python
# Ensure model is on GPU
model = model.to("cuda")

# Use half precision
model = model.half()  # Same as torch.float16

# Use fewer tokens
outputs = model.generate(..., max_new_tokens=50)  # Less work = faster

# Enable SDPA attention (faster attention mechanism)
from torch.nn.functional import scaled_dot_product_attention

# For newer models, use optimized attention
model.config.attn_implementation = "flash_attention_2"
```

---

## Conclusion and Next Steps

You now understand:
✅ How Large Language Models work
✅ Different quantization strategies
✅ Multiple frameworks for local inference
✅ How to build a simple chatbot
✅ Performance benchmarking
✅ Practical applications for biology

**Key Takeaways:**
- Start with small models (7B) for learning
- Use quantization for memory efficiency
- Balance quality vs. speed based on your needs
- Local inference ensures privacy

**Next Notebook:**
In Notebook 3, we'll learn about embeddings and semantic search - essential for retrieval-augmented generation (RAG) systems that let LLMs access external knowledge!
