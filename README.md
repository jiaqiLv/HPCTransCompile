# HPCTransCompile

## Introduction

This repository contains the official implementation of **HPCTransCompile: An AI Compiler Generated Dataset for High-Performance CUDA Transpilation and LLM Preliminary Exploration.**

## Framework Implementation

![framework](https://github.com/user-attachments/assets/2cb2db60-e95a-4e59-b949-98a01fc86acd)

We propose a novel framework for generating high-performance CUDA and corresponding platform code pairs, leveraging AI compiler and automatic optimization technology. We further enhance the framework with a graph-based data augmentation method and introduce HPCTransEval, a benchmark for evaluating LLM performance on CUDA transpilation. We conduct experiments using CUDA-to-CPU transpilation as a case study on leading LLMs. The result demonstrates that our framework significantly improves CUDA transpilation, highlighting the potential of LLMs to address compatibility challenges within the CUDA ecosystem.

## Setup

We provide two benchmarks, `HPCTransEval` and `KernelBench_c`. You can find them in the corresponding folders. In order to do the assessment correctly, you need to download a modified library `tvm` from https://github.com/hehesnail/tvm/tree/modify_style. Then, run:

```
pip install -r requirements.txt
```

1. To evaluate on `HPCTransEval`, run:
   
   ```
   bash EvalEngine/eval_HPCTransEval.sh
   ```
2. To evaluate on `KernelBench_c`, run:

   ```
   bash EvalEngine/eval_KernelBench_c.sh
   ```
