# Local AI for Biology — Workshop

This repository contains materials for a hands-on workshop: "Local AI for Biology".
Target audience: beginners and intermediate practitioners in biology / bioinformatics who want to run and experiment with local LLMs and related tooling using Jupyter notebooks.

Structure
- Course/: markdown-driven notebooks and guides (the `01-setup-env.md` file lives here)
- notebooks/: Jupyter notebooks (teaching and exercises)
- docs/: workshop scope, learning goals, and additional docs
- labs/: hands-on lab instructions and datasets
- code/: example code and microservices (FastAPI/Streamlit demos)
- scripts/: helper and verification scripts

Delivery format
- Markdown guides + Jupyter notebooks (notebooks are runnable and intended to be executed inside a conda/mamba environment)

How to get started (Linux)
1. Open a terminal and review `Course/01-setup-env.md`.
2. Create a conda/mamba environment (recommended: Mambaforge).
3. Run `scripts/verify_setup.sh` to quickly check key components.

See `docs/WORKSHOP_SCOPE.md` for the workshop scope and module plan.
