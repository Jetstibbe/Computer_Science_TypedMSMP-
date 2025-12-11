# Typed-MSMP+ extension with LSH Blocking for Duplicate Product Detection (CS Individual Assignment)

This repository contains the code for a duplicate detection project on TV
products, based on the MSMP+ framework and typed numerical features. The
implementation is used to generate the results and figures reported in the
accompanying paper.

## Project overview

The goal of the project is to detect duplicate TV products across multiple
Qeb shops. Each product is represented by model word and attribute based
tokens, which are then enriched with typed measurement labels (e.g. inch,
hertz, resolution, brightness). MinHash and Locality-Sensitive Hashing (LSH)
are used as a blocking strategy to reduce the number of pairwise comparisons
before applying MSM style clustering.

Three variants are implemented and compared:

- **MSMP+**: baseline using untyped numeric and alphanumeric model words.
- **Typed-MSMP+**: typed numeric tokens plus untyped title model words.
- **Combined-MSMP+**: union of MSMP+ and Typed-MSMP+ token sets.

All three variants share the same downstream pipeline (MinHash, LSH,
MSM-based clustering).

## Repository structure

The main Python modules are:

- `msmp_typed_main.py`  
  Main pipeline for building token representations,
  computing MinHash signatures, running LSH blocking and applying MSM clustering,. 
  Also contains evaluation routines on the labelled test set.

- `msmp_typed_variants.py`  
  Construction of the three token variants (MSMP+, Typed-MSMP+, Combined).
  Implements the tokenisation and typing logic for titles and attribute
  values, including group handling for patterns such as `A x B` and
  unit-aware rounding.

- `encoder_typed.py`  
  Vocabulary and encoder classes. Maps string tokens to integer feature
  IDs and provides encoders for title model words and typed model word
  tokens.

- `minhash_typed.py`  
  Implementation of MinHash signatures for sets of feature IDs. Generates
  linear hash functions of the form `h(x) = (a * x + b) mod p` and
  computes signatures of fixed length.

- `lsh_typed.py`  
  LSH banding on MinHash signatures. Supports both exact layouts
  (`bands * rows_per_band = signature_length`) and the plus-one layout
  used in the experiments. Produces candidate pairs from bucket
  collisions.

- `msmp_typed_bootstrap_grids.py`  
  Scripts for parameter search and bootstrap evaluation.
  Explores combinations of LSH parameters `(r, b)` and MSM thresholds
  `ε` and writes out result files (e.g. PC, PQ, F1*).

- `msmp_typed_plot.py`  
  Utilities for reading result files and generating the figures used in
  the paper (e.g. PC vs. PQ curves, clustering F1 plots).

## Requirements

The code is written in Python. Typical dependencies are:

- `numpy`
- `pandas`
- `matplotlib`
- `typing`
- `math`

## How to use the code

A typical usage pattern is:

1. Prepare the TV product data (titles, attributes, labels) and put it in
   a `data/` directory.
2. Either run the pipeline script (e.g. `msmp_typed_main.py`) to construct
   representations, apply MinHash + LSH, and perform MSM clustering.
3. Or use  `msmp_typed_bootstrap_grids.py` and `msmp_typed_plot.py` to
   reproduce grid searches and plots reported in the paper.

This description is sufficient for reproducing the experiments
described in the paper. For transparancy purposes, the GitHub URL of this
repository is cited in the paper.
