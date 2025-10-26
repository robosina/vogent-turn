# Vogent Turn - Benchmark Report

**Date:** October 26, 2025  
**Model:** vogent/Vogent-Turn-80M  
**Repository:** https://github.com/vogent/vogent-turn

---

## Executive Summary

**Performance Highlights:**
- ⚡ **5ms single inference latency** (P95: 5.94ms)
- 🚀 **283 predictions/second** with optimal batching (batch_size=32)
- 💾 **4-5% GPU memory** usage (~1GB on 23GB GPU)
- 📊 **1.63× speedup** with batch processing
- ✅ **Production-ready** performance with low latency variance

**System:**
- GPU: NVIDIA A10G (23 GB)
- CUDA: 12.4
- PyTorch: 2.9.0+cu128

---

## System Configuration

### Hardware
- **GPU:** NVIDIA A10G
- **GPU Memory:** 23,028 MiB (22.5 GB)
- **GPU Driver:** 550.127.05
- **CUDA Version:** 12.4

### Software
- **OS:** Linux 6.8.0-1019-aws (Ubuntu)
- **Conda:** 24.11.0
- **Python:** 3.10.19
- **PyTorch:** 2.9.0+cu128
- **Transformers:** 4.57.1

---

## Installation Guide

### 1. Create Conda Environment

```bash
cd /path/to/vogent-turn
conda create -n vogent-turn python=3.10 -y
```

### 2. Install Package

```bash
conda activate vogent-turn
pip install -e .
```

### 3. Install Benchmark Dependencies

```bash
pip install nvidia-ml-py3 matplotlib
```

### 4. Fix BFloat16 Compatibility Issue

The package requires a small fix for BFloat16 to NumPy conversion. Edit `vogent_turn/inference.py`:

**Line 294:**
```python
# Change from:
probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

# To:
probs = torch.softmax(logits, dim=1)[0].float().cpu().numpy()
```

**Line 433:**
```python
# Change from:
probs = torch.softmax(logits, dim=1).cpu().numpy()

# To:
probs = torch.softmax(logits, dim=1).float().cpu().numpy()
```

### 5. Authenticate with Hugging Face

The model is gated and requires authentication:

```bash
huggingface-cli login
# Or set environment variable:
export HF_TOKEN="your_token_here"
```

Then request access at: https://huggingface.co/vogent/Vogent-Turn-80M

---

## Benchmark Results

### Single Inference Performance

| Metric | Value |
|--------|-------|
| **Mean Latency** | 5.01 ms |
| **Median Latency** | 4.94 ms |
| **P95 Latency** | 5.94 ms |
| **P99 Latency** | 6.08 ms |
| **Min Latency** | 4.52 ms |
| **Max Latency** | 6.16 ms |
| **Throughput** | 199.5 predictions/sec |
| **GPU Utilization** | 39.4% (mean), 47.0% (peak) |
| **GPU Memory Usage** | 4.1% (~940 MB) |

#### GPU Utilization Over Time (Single Inference)

![GPU Utilization](gpu_utilization.png)

The chart above shows GPU utilization and memory usage during 100 consecutive inference runs. The model maintains consistent GPU utilization around 39% with minimal memory footprint.

### Batch Processing Performance

| Batch Size | Total Time (ms) | Per Sample (ms) | Throughput (preds/s) | Speedup | GPU Util (%) | GPU Memory (%) |
|------------|-----------------|-----------------|----------------------|---------|--------------|----------------|
| 1          | 5.76           | 5.76            | 173.8                | 1.00×   | 33.3         | 4.1            |
| 2          | 8.52           | 4.26            | 234.9                | 1.35×   | 34.0         | 4.1            |
| 4          | 17.71          | 4.43            | 225.9                | 1.30×   | 33.6         | 4.1            |
| 8          | 30.89          | 3.86            | 259.0                | 1.49×   | 33.5         | 4.2            |
| 16         | 58.63          | 3.66            | 272.9                | 1.57×   | 33.4         | 4.5            |
| **32**     | **112.98**     | **3.53**        | **283.2**            | **1.63×** | **33.9**   | **5.4**        |

#### Batch Processing Performance Analysis

![Batch Performance Analysis](batch_performance.png)

The comprehensive analysis above shows 6 key metrics across different batch sizes:

1. **Total Batch Time** - Shows expected linear scaling with batch size
2. **Per-Sample Time** - Demonstrates efficiency gains from batching (5.76ms → 3.53ms)
3. **Throughput** - Peaks at batch size 32 with 283 predictions/second
4. **Speedup** - Achieves 1.63× speedup compared to single inference
5. **GPU Utilization** - Remains consistent at 33-34% across all batch sizes (mean and peak)
6. **GPU Memory Usage** - Minimal growth from 4.1% to 5.4% even at batch size 32

### Model Initialization

- **Compilation Time:** ~26 seconds (with `torch.compile` and warmup)
- **First Run Overhead:** Includes Triton kernel autotuning

---

## Key Findings

### Performance
✅ **Excellent Latency:** ~5ms single inference, ~3.5ms with batching  
✅ **High Throughput:** Up to 283 predictions/second with batch_size=32  
✅ **Consistent Performance:** Low variance across runs (±0.43ms std)  
✅ **Efficient Batching:** 1.63× speedup with optimal batching

### GPU Efficiency
✅ **Low Memory Footprint:** Only 4-5% GPU memory (~1GB)  
✅ **Moderate Utilization:** 33-40% GPU utilization  
⚠️ **Headroom Available:** GPU underutilized - can run multiple instances

### Prediction Quality Example
```
Sample 1: "What is your phone number" → "My number is 804"
├─ Turn Complete: False
├─ Endpoint Probability: 25.8%
└─ Continue Probability: 74.2%

Sample 2: "What is your phone number" → "My number is 804 555 1234"
├─ Turn Complete: True
├─ Endpoint Probability: 89.5%
└─ Continue Probability: 10.7%
```

---

## Recommendations

### For Production Deployment

1. **Low Latency Use Case (Real-time Voice AI)**
   - Use **batch_size=8** 
   - Latency: ~3.86ms per sample
   - Throughput: 259 predictions/sec

2. **High Throughput Use Case (Batch Processing)**
   - Use **batch_size=32**
   - Latency: ~3.53ms per sample
   - Throughput: 283 predictions/sec

3. **Resource Optimization**
   - GPU memory usage is minimal (4-5%)
   - Can run **4-5 parallel instances** on this GPU
   - Consider multi-instance deployment for higher throughput

### Optimization Opportunities

1. **Increase GPU Utilization**
   - Current: 33-40%
   - Potential: Multi-model serving, larger batches

2. **TensorRT Optimization**
   - Consider TensorRT conversion for further speedup
   - Potential 2-3× additional performance gain

3. **Mixed Precision**
   - Already using BFloat16
   - Consider FP16 for even faster inference

---

## Running Benchmarks

### Basic Benchmark (Single Inference + GPU Monitoring)
```bash
conda activate vogent-turn
python benchmark.py
```
**Output:** Console metrics + `gpu_utilization.png`

### Batch Processing Benchmark
```bash
conda activate vogent-turn
python test_batch_processing.py
```
**Output:** Console metrics + `batch_performance.png`

### Simple Test
```bash
conda activate vogent-turn
python run.py
```

---

## Generated Files & Visualizations

### Benchmark Scripts
- **`benchmark.py`** - Single inference benchmark with GPU monitoring
- **`test_batch_processing.py`** - Comprehensive batch processing analysis
- **`run.py`** - Simple usage example

### Performance Visualizations

#### 1. GPU Utilization Chart (`gpu_utilization.png`)
Generated by `benchmark.py` - Shows real-time GPU metrics during 100 inference runs:
- Top panel: GPU utilization over time with mean indicator
- Bottom panel: GPU memory usage over time with mean indicator

![GPU Utilization Chart](gpu_utilization.png)

#### 2. Batch Performance Analysis (`batch_performance.png`)
Generated by `test_batch_processing.py` - Comprehensive 6-panel analysis:
- **Total Batch Time:** How total processing time scales with batch size
- **Per-Sample Time:** Efficiency gains from batching
- **Throughput:** Predictions per second across batch sizes
- **Speedup:** Performance improvement vs. single inference
- **GPU Utilization:** Mean and peak GPU usage by batch size
- **GPU Memory Usage:** Memory footprint scaling

![Batch Performance Chart](batch_performance.png)

---

## Model Architecture

- **Audio Encoder:** Whisper-Tiny (processes up to 8 seconds @ 16kHz)
- **Text Model:** SmolLM-135M (12 layers, ~80M parameters)
- **Classifier:** Binary (turn complete / turn incomplete)
- **Input:** Multimodal (audio + text context)

---

## Dependencies

Core dependencies installed in conda environment:
```
torch>=2.1.0
transformers>=4.35.0
numpy>=1.24.0
huggingface-hub>=0.19.0
soundfile>=0.12.0
librosa>=0.10.0
resampy>=0.4.2
cffi>=1.15.0
```

Additional for benchmarking:
```
nvidia-ml-py3
matplotlib
```

---

## Notes

1. **torch.compile Optimization:** The first initialization takes ~26s due to Triton kernel compilation and warmup. Subsequent runs are fast.

2. **BFloat16 Handling:** NumPy doesn't support BFloat16 natively, requiring conversion to float32 before `.numpy()`.

3. **Gated Model:** Requires Hugging Face authentication and access approval.

4. **GPU Compatibility:** Requires CUDA-capable GPU. Tested on NVIDIA A10G with CUDA 12.4.

---

## Contact & References

- **Model Page:** https://huggingface.co/vogent/Vogent-Turn-80M
- **Technical Report:** https://blog.vogent.ai/posts/voturn-80m-state-of-the-art-turn-detection-for-voice-agents
- **Demo:** https://huggingface.co/spaces/vogent/vogent-turn-demo
- **Repository:** https://github.com/vogent/vogent-turn

---

**Report Generated:** October 26, 2025  
**Benchmark Environment:** AWS EC2 instance with NVIDIA A10G GPU

