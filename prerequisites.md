# Workshop Prerequisites: A Guide to Setting Up Your Local AI Development Environment

Welcome to the workshop! To ensure we can dive straight into the practical sessions without delays, please follow this guide to install the necessary software and libraries on your laptop **at least two days before the event**.

This guide explains what each tool does, why it's important, and how to install and configure it correctly.

**Estimated Time & Disk Space:**
*   **Time:** 30–60 minutes, depending on your internet speed.
*   **Disk Space:** Approximately 15–20 GB for all tools and models.

---

### Quick Setup Checklist

For experienced users who just need to verify their setup, here is a quick checklist.

- [ ] **Python 3.9+** with `pip` or `conda`.
- [ ] **Windsurf IDE**: The primary code editor for the workshop.
- [ ] **PyTorch 2.4+**: Core deep learning framework (latest CUDA version).
- [ ] **Hugging Face Libraries**: `transformers`, `accelerate`, `bitsandbytes`.
- [ ] **Ollama**: Local LLM runner.
- [ ] **Ollama Models**: `ollama pull phi3` and `ollama pull llama3`.
- [ ] **TabbyML**: Self-hosted code assistant (Docker recommended).
- [ ] **Git & Command-Line Tools**: For version control and terminal access.

---

### 1. Understanding Your Toolkit: What and Why

Here’s a breakdown of the software you will be installing. Understanding their roles will help you see how they fit together.

*   **Python (version 3.9 or newer)**
    *   **What it is:** A versatile programming language that is the standard for machine learning.
    *   **Why we need it:** All our scripts and libraries are Python-based. You'll use `pip` (Python's package installer) to manage dependencies.
    *   **Official Docs:** [python.org](https://www.python.org/ )

*   **Windsurf**
    *   **What it is:** An AI-powered IDE based on VS Code, designed for advanced coding assistance.
    *   **Why we need it:** It provides powerful, context-aware code generation and a seamless interface for interacting with local models.
    *   **Official Docs:** [windsurf.ai](https://windsurf.ai/docs )

*   **PyTorch**
    *   **What it is:** A powerful, open-source machine learning framework.
    *   **Why we need it:** It is the engine that executes our AI models. **Transformers** provides the model architectures, **Accelerate** helps them run efficiently, and **PyTorch** performs the underlying calculations.
    *   **Official Docs:** [pytorch.org](https://pytorch.org/ )

*   **Hugging Face Libraries (`transformers`, `accelerate`, `bitsandbytes`)**
    *   **What they are:** A suite of libraries to download, run, and optimize state-of-the-art models.
    *   **Why we need them:** `transformers` gives us the models, `accelerate` optimizes their execution, and `bitsandbytes` reduces memory usage on GPUs.
    *   **Official Docs:** [Hugging Face Docs](https://huggingface.co/docs )

*   **Ollama**
    *   **What it is:** A tool that lets you easily run large language models (LLMs) locally.
    *   **Why we need it:** It provides a simple command-line interface and an API to interact with powerful models privately on your machine.
    *   **Official Docs:** [ollama.com](https://ollama.com/ )

*   **TabbyML**
    *   **What it is:** A self-hosted AI coding assistant, like a private GitHub Copilot.
    *   **Why we need it:** It provides intelligent code completions directly in your editor while keeping your code private.
    *   **Official Docs:** [tabby.tabbyml.com](https://tabby.tabbyml.com/ )

*   **Git / Command-Line Tools**
    *   **What they are:** Git is for version control; command-line tools are for interacting with your system.
    *   **Why we need them:** We'll use Git to access workshop code and the command line for all installations.
    *   **Official Docs:** [git-scm.com](https://git-scm.com/ )

---

### 2. Hardware & Driver Notes

*   **CPU vs. GPU:** All tools will work on a modern CPU. However, performance is significantly better with a dedicated GPU.
*   **NVIDIA GPU Users:** For the best experience, please **update your GPU drivers** to the latest version available (e.g., version 550 or newer is recommended for full CUDA 12.x compatibility). You can check your driver version with the `nvidia-smi` command.
*   **Apple Silicon (M1/M2/M3) Users:** No extra drivers are needed. PyTorch and other libraries will automatically use Apple's Metal Performance Shaders (MPS) for acceleration. All listed tools have native ARM64 support.

---

### 3. Installation Instructions

Please follow the instructions for your operating system.

#### macOS (Apple Silicon & Intel)

1.  **Install Windsurf:** Download and install from the [official Windsurf website](https://windsurf.ai/ ).
2.  **Install Homebrew (if not already installed):** Open the Windsurf terminal (`Terminal > New Terminal`) and run:
    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh )"
    ```
3.  **Install Core Tools:**
    ```bash
    brew install python ollama
    ```
4.  **Create and Activate a Virtual Environment:**
    ```bash
    python3 -m venv workshop_env && source workshop_env/bin/activate
    ```
5.  **Install Python Libraries:**
    ```bash
    pip install --upgrade pip
    pip install torch torchvision torchaudio
    pip install "transformers==4.41.2" "accelerate==0.30.1" "bitsandbytes==0.43.1"
    ```
6.  **Install TabbyML (Docker Recommended):**
    *First, install Docker Desktop for Mac. Then run:*
    ```bash
    docker run -it -p 8080:8080 -v ~/.tabby:/data tabbyml/tabby serve --model DeepSeek-Coder-V2-Lite-Instruct
    ```
7.  **Pull Ollama Models:**
    ```bash
    ollama pull phi3 && ollama pull llama3
    ```

#### Linux (Debian/Ubuntu)

1.  **Install Windsurf:** Download and install the `.deb` or AppImage file from the [official Windsurf website](https://windsurf.ai/ ).
2.  **Update and Install Dependencies:** Open the Windsurf terminal and run:
    ```bash
    sudo apt update && sudo apt install -y git curl build-essential python3-venv python3-pip
    ```
3.  **Install Ollama:**
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```
4.  **Install Docker and TabbyML:**
    *   **Install Docker:**
        ```bash
        sudo apt install -y docker.io
        sudo systemctl start docker && sudo systemctl enable docker
        sudo usermod -aG docker $USER # Requires re-login to take effect
        ```
    *   **Run TabbyML:**
        ```bash
        docker run -it --gpus all -p 8080:8080 -v $HOME/.tabby:/data tabbyml/tabby serve --model DeepSeek-Coder-V2-Lite-Instruct
        ```
5.  **Create and Activate a Virtual Environment:**
    ```bash
    python3 -m venv workshop_env && source workshop_env/bin/activate
    ```
6.  **Install Python Libraries:**
    ```bash
    pip install --upgrade pip
    # For NVIDIA GPU users (with updated drivers ):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    # For CPU-only users:
    # pip install torch torchvision torchaudio
    pip install "transformers==4.41.2" "accelerate==0.30.1" "bitsandbytes==0.43.1"
    ```
7.  **Pull Ollama Models:**
    ```bash
    ollama pull phi3 && ollama pull llama3
    ```

#### Windows

1.  **Install Windsurf:** Download and install from the [official Windsurf website](https://windsurf.ai/ ).
2.  **Install WSL2:** If you don't have it, follow the [Official Microsoft WSL2 Installation Guide](https://learn.microsoft.com/en-us/windows/wsl/install ). This is highly recommended.
3.  **Install Python:** Download and install Python 3.9+ from the [official Python website](https://www.python.org/downloads/ ). **Important:** Check the box that says "Add Python to PATH" during installation.
4.  **Install Ollama:** Download and run the installer from the [official Ollama website](https://ollama.com ).
5.  **Install Docker and TabbyML (inside WSL2):**
    *   Open your WSL2 terminal (e.g., Ubuntu).
    *   Follow the Linux instructions above to install Docker and run TabbyML.
6.  **Create and Activate a Virtual Environment (in PowerShell or CMD):**
    ```powershell
    python -m venv workshop_env
    .\workshop_env\Scripts\activate
    ```
7.  **Install Python Libraries:**
    ```bash
    pip install --upgrade pip
    # For NVIDIA GPU users (with updated drivers):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    # For CPU-only users:
    # pip install torch torchvision torchaudio
    pip install "transformers==4.41.2" "accelerate==0.30.1"
    # bitsandbytes on Windows can be tricky. Try this, but skip if it fails:
    pip install bitsandbytes
    ```
8.  **Pull Ollama Models (in PowerShell or CMD ):**
    ```bash
    ollama pull phi3 && ollama pull llama3
    ```

---

### 4. Configuring Your Environment

An essential step is to tell your IDE (Windsurf) and other tools to use the Python interpreter from your virtual environment.

#### Configuring Windsurf

1.  **Open Your Project Folder:** Launch Windsurf and open the main folder that contains the `workshop_env` directory.
2.  **Select the Python Interpreter:**
    *   Windsurf should automatically detect your virtual environment. Check the bottom-right corner of the status bar. It should say something like **"Python 3.10.x ('workshop_env')"**.
    *   If not, click on the interpreter version, select **"Select Interpreter"** from the command palette, and choose the one pointing to your `workshop_env`.

#### Configuring Hugging Face Accelerate (Optional)

For fine-grained control over hardware usage (e.g., CPU vs. GPU, mixed precision), run this one-time configuration:
```bash
accelerate config
### 5. Post-Install Verification

Run these commands **inside the Windsurf terminal** (with your `workshop_env` activated) to ensure everything is working correctly.

1.  **Check Python Environment and Libraries:**
    ```python
    import sys
    import torch
    import transformers
    import accelerate

    print(f"Python executable: {sys.executable}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"Accelerate version: {accelerate.__version__}")
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        print("Apple Metal (MPS) GPU detected.")
    else:
        print("No GPU detected. Running on CPU.")
    ```
    *   **Expected Output:** The path should point to your `workshop_env`, library versions should be printed, and your hardware (GPU or CPU) should be correctly identified.

2.  **Test Ollama Service:**
    ```bash
    ollama run phi3 "Hello! Who are you?"
    ```
    *   **Expected Output:** A streaming response from the Phi-3 model.

3.  **Test Hugging Face Transformers:**
    ```python
    from transformers import pipeline
    generator = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print(generator("The future of AI is", max_new_tokens=20))
    ```
    *   **Expected Output:** A short, generated sentence starting with "The future of AI is...".
### 6. Troubleshooting Tips

If you run into issues, here are some common fixes.

*   **`pip install` fails with "Permission Denied":**
    *   **Fix:** Ensure you are in an **activated virtual environment**. The terminal prompt should start with `(workshop_env)`. Avoid using `sudo pip`.

*   **`bitsandbytes` installation fails:**
    *   **Fix:** This is common on Windows or with unsupported hardware. You can safely skip it by running `pip uninstall bitsandbytes` and proceeding. The workshop will still be fully functional.

*   **CUDA-related errors (e.g., "CUDA not found"):**
    *   **Fix (NVIDIA users):** This usually means a driver mismatch. Ensure your GPU drivers are up to date (version 550+ recommended). The PyTorch installation command with `cu124` is aligned with the latest drivers.

*   **Ollama connection error (`ECONNREFUSED`):**
    *   **Fix:** Make sure the Ollama application is running in the background. On macOS and Windows, you should see its icon in the system tray/menu bar. On Linux, you can run `ollama serve`.

*   **Docker command fails with "permission denied":**
    *   **Fix (Linux):** You may need to re-login after adding your user to the `docker` group (`sudo usermod -aG docker $USER`). Alternatively, prefix Docker commands with `sudo`.
