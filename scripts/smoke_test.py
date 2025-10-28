#!/usr/bin/env python3
"""Robust headless smoke test for the workshop environment.

This script resolves repo paths relative to the scripts/ directory and supports
an option to skip generation to avoid model downloads in CI or offline runs.
"""
import sys
import subprocess
import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description='Workshop smoke test')
    parser.add_argument('--skip-generation', action='store_true', help='Skip transformer generation step (avoids model downloads)')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / 'labs' / 'data'

    errors = []
    print('Python', sys.version)

    # Check imports
    modules = [
        'transformers',
        'sentence_transformers',
        'sklearn',
        'Bio',
    ]
    for name in modules:
        try:
            __import__(name)
            print(f'OK import {name}')
        except Exception as e:
            print(f'FAIL import {name}: {e}')
            errors.append(f'import:{name}')

    # Read sample files
    txt = ''
    try:
        with open(data_dir / 'sample_docs.txt') as f:
            txt = f.read()
        print('Read sample_docs.txt length=', len(txt))
    except Exception as e:
        print('FAIL read sample_docs.txt', e)
        errors.append('read:sample_docs')

    try:
        with open(data_dir / 'sample_fasta.fasta') as f:
            fasta = f.read()
        print('Read sample_fasta.fasta length=', len(fasta))
    except Exception as e:
        print('FAIL read sample_fasta.fasta', e)
        errors.append('read:sample_fasta')

    # Tiny generation using transformers pipeline (skip if requested)
    if args.skip_generation:
        print('SKIP generation step (user requested skip)')
    else:
        try:
            from transformers import pipeline
            gen = pipeline('text-generation', model='distilgpt2')
            out = gen('Translate to short: What is CRISPR?', max_length=50, num_return_sequences=1)
            print('Generator OK — sample output:', out[0]['generated_text'][:120])
        except Exception as e:
            print('SKIP generation step (transformers) —', e)
            errors.append('generation')

    # TF-IDF vectorization test
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(stop_words='english')
        docs = txt.split('\n\n')[:3]
        X = vec.fit_transform(docs)
        print('TF-IDF shape', X.shape)
    except Exception as e:
        print('SKIP TF-IDF test —', e)
        errors.append('tfidf')

    # Biopython parse test
    try:
        from Bio import SeqIO
        records = list(SeqIO.parse(data_dir / 'sample_fasta.fasta', 'fasta'))
        print('Parsed FASTA records:', len(records))
    except Exception as e:
        print('SKIP Biopython parse —', e)
        errors.append('biopython')

    # NVIDIA availability
    try:
        r = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode == 0:
            print('nvidia-smi available')
        else:
            print('nvidia-smi not available')
    except Exception:
        print('nvidia-smi not found')

    print('\nSummary:')
    if errors:
        print('WARN — some checks failed or were skipped:', errors)
        return 2
    else:
        print('PASS — all checks OK')
        return 0


if __name__ == '__main__':
    sys.exit(main())
