import time
import numpy as np
from vogent_turn import TurnDetector
import soundfile as sf
import urllib.request
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import threading

try:
    import pynvml
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: pynvml not installed. GPU monitoring disabled.")

class GPUMonitor:
    """Monitor GPU utilization in a background thread"""
    def __init__(self, device_id=0, sample_interval=0.01):
        self.device_id = device_id
        self.sample_interval = sample_interval
        self.monitoring = False
        self.gpu_utils = []
        self.gpu_memory = []
        self.timestamps = []
        self.thread = None
        
        if GPU_AVAILABLE:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        start_time = time.time()
        while self.monitoring:
            if GPU_AVAILABLE:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                    
                    self.timestamps.append(time.time() - start_time)
                    self.gpu_utils.append(util.gpu)
                    self.gpu_memory.append(memory_info.used / memory_info.total * 100)
                except Exception as e:
                    pass
            time.sleep(self.sample_interval)
    
    def start(self):
        """Start monitoring"""
        if not GPU_AVAILABLE:
            return
        self.monitoring = True
        self.gpu_utils = []
        self.gpu_memory = []
        self.timestamps = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def get_stats(self):
        """Get statistics"""
        if not self.gpu_utils:
            return None
        return {
            'mean_util': np.mean(self.gpu_utils),
            'max_util': np.max(self.gpu_utils),
            'mean_memory': np.mean(self.gpu_memory),
            'max_memory': np.max(self.gpu_memory),
        }
    
    def __del__(self):
        """Cleanup"""
        if GPU_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

print("=" * 70)
print("Vogent Turn - Batch Processing Test")
print("=" * 70)

# Download sample audio files
print("\n1. Downloading sample audio files...")
audio_urls = [
    "https://storage.googleapis.com/voturn-sample-recordings/incomplete_number_sample.wav",
    "https://storage.googleapis.com/voturn-sample-recordings/complete_number_sample.wav",
]

audio_samples = []
contexts = []

# Download first sample
print("   Downloading sample 1...")
urllib.request.urlretrieve(audio_urls[0], "sample1.wav")
audio1, sr1 = sf.read("sample1.wav")
audio_samples.append(audio1)
contexts.append({
    'prev_line': "What is your phone number",
    'curr_line': "My number is 804"
})
print(f"   ✓ Sample 1: {len(audio1)} samples at {sr1}Hz ({len(audio1)/sr1:.2f}s)")

# Try to download second sample (may or may not exist)
try:
    print("   Downloading sample 2...")
    urllib.request.urlretrieve(audio_urls[1], "sample2.wav")
    audio2, sr2 = sf.read("sample2.wav")
    audio_samples.append(audio2)
    contexts.append({
        'prev_line': "What is your phone number",
        'curr_line': "My number is 804 555 1234"
    })
    print(f"   ✓ Sample 2: {len(audio2)} samples at {sr2}Hz ({len(audio2)/sr2:.2f}s)")
except Exception as e:
    print(f"   ⚠ Sample 2 not available, will duplicate sample 1")
    audio_samples.append(audio1)
    contexts.append({
        'prev_line': "How are you today",
        'curr_line': "I'm doing great"
    })

# Initialize detector
print("\n2. Initializing TurnDetector...")
print("   (This may take ~30 seconds for torch.compile optimization)")
start_init = time.time()
detector = TurnDetector(compile_model=True, warmup=True)
init_time = time.time() - start_init
print(f"   ✓ Initialization complete: {init_time:.2f}s")

# Test different batch sizes
print("\n3. Testing different batch sizes...")
print("-" * 70)

batch_sizes = [1, 2, 4, 8, 16, 32]
results = {
    'batch_size': [],
    'total_time_ms': [],
    'per_sample_ms': [],
    'throughput': [],
    'gpu_util_mean': [],
    'gpu_util_max': [],
    'gpu_memory_mean': []
}

for batch_size in batch_sizes:
    # Create batch by repeating samples
    audio_batch = []
    context_batch = []
    for i in range(batch_size):
        audio_batch.append(audio_samples[i % len(audio_samples)])
        context_batch.append(contexts[i % len(contexts)])
    
    # Warmup
    for _ in range(3):
        _ = detector.predict_batch(
            audio_batch,
            context_batch=context_batch,
            sample_rate=sr1,
            return_probs=False
        )
    
    # Benchmark with GPU monitoring
    num_iterations = 20
    times = []
    
    gpu_monitor = GPUMonitor()
    gpu_monitor.start()
    
    for _ in range(num_iterations):
        start = time.time()
        batch_results = detector.predict_batch(
            audio_batch,
            context_batch=context_batch,
            sample_rate=sr1,
            return_probs=True
        )
        elapsed = (time.time() - start) * 1000  # ms
        times.append(elapsed)
    
    gpu_monitor.stop()
    gpu_stats = gpu_monitor.get_stats()
    
    mean_time = np.mean(times)
    per_sample_time = mean_time / batch_size
    throughput = 1000 / per_sample_time
    
    results['batch_size'].append(batch_size)
    results['total_time_ms'].append(mean_time)
    results['per_sample_ms'].append(per_sample_time)
    results['throughput'].append(throughput)
    
    if gpu_stats:
        results['gpu_util_mean'].append(gpu_stats['mean_util'])
        results['gpu_util_max'].append(gpu_stats['max_util'])
        results['gpu_memory_mean'].append(gpu_stats['mean_memory'])
    else:
        results['gpu_util_mean'].append(0)
        results['gpu_util_max'].append(0)
        results['gpu_memory_mean'].append(0)
    
    print(f"Batch Size {batch_size:2d}:")
    print(f"  Total time: {mean_time:6.2f} ms")
    print(f"  Per sample: {per_sample_time:6.2f} ms")
    print(f"  Throughput: {throughput:6.1f} predictions/sec")
    if gpu_stats:
        print(f"  GPU Util:   {gpu_stats['mean_util']:6.1f}% (peak: {gpu_stats['max_util']:.1f}%)")
        print(f"  GPU Memory: {gpu_stats['mean_memory']:6.1f}%")
    print()

# Test with detailed results
print("\n4. Detailed batch prediction example...")
print("-" * 70)

# Create a diverse batch
test_batch = audio_samples[:2] if len(audio_samples) > 1 else [audio_samples[0], audio_samples[0]]
test_contexts = [
    {'prev_line': "What is your phone number", 'curr_line': "My number is 804"},
    {'prev_line': "What is your phone number", 'curr_line': "My number is 804 555 1234"},
]

print(f"Processing batch of {len(test_batch)} samples...\n")

start = time.time()
batch_results = detector.predict_batch(
    test_batch,
    context_batch=test_contexts,
    sample_rate=sr1,
    return_probs=True
)
elapsed = (time.time() - start) * 1000

for i, result in enumerate(batch_results):
    print(f"Sample {i+1}:")
    print(f"  Context: \"{test_contexts[i]['prev_line']}\" → \"{test_contexts[i]['curr_line']}\"")
    print(f"  Turn complete: {result['is_endpoint']}")
    print(f"  Endpoint prob: {result['prob_endpoint']:.1%}")
    print(f"  Continue prob: {result['prob_continue']:.1%}")
    print()

print(f"Batch processing time: {elapsed:.2f} ms")
print(f"Average per sample: {elapsed/len(test_batch):.2f} ms")

# Plot results
print("\n5. Generating performance charts...")
print("-" * 70)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[2, 0])
ax6 = fig.add_subplot(gs[2, 1])

# Plot 1: Total batch time vs batch size
ax1.plot(results['batch_size'], results['total_time_ms'], 'b-o', linewidth=2, markersize=8)
ax1.set_xlabel('Batch Size', fontsize=12)
ax1.set_ylabel('Total Batch Time (ms)', fontsize=12)
ax1.set_title('Total Batch Processing Time', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log', base=2)

# Plot 2: Per-sample time vs batch size
ax2.plot(results['batch_size'], results['per_sample_ms'], 'g-o', linewidth=2, markersize=8)
ax2.set_xlabel('Batch Size', fontsize=12)
ax2.set_ylabel('Time per Sample (ms)', fontsize=12)
ax2.set_title('Per-Sample Processing Time', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log', base=2)

# Plot 3: Throughput vs batch size
ax3.plot(results['batch_size'], results['throughput'], 'r-o', linewidth=2, markersize=8)
ax3.set_xlabel('Batch Size', fontsize=12)
ax3.set_ylabel('Throughput (predictions/sec)', fontsize=12)
ax3.set_title('Inference Throughput', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xscale('log', base=2)

# Plot 4: Speedup vs batch size
baseline = results['per_sample_ms'][0]  # batch_size=1
speedups = [baseline / t for t in results['per_sample_ms']]
ax4.plot(results['batch_size'], speedups, 'm-o', linewidth=2, markersize=8)
ax4.axhline(y=1.0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (batch=1)')
ax4.set_xlabel('Batch Size', fontsize=12)
ax4.set_ylabel('Speedup (×)', fontsize=12)
ax4.set_title('Batch Processing Speedup', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.set_xscale('log', base=2)

# Plot 5: GPU Utilization vs batch size
if GPU_AVAILABLE and any(results['gpu_util_mean']):
    ax5.plot(results['batch_size'], results['gpu_util_mean'], 'orange', marker='o', linewidth=2, markersize=8, label='Mean')
    ax5.plot(results['batch_size'], results['gpu_util_max'], 'red', marker='s', linewidth=2, markersize=6, linestyle='--', label='Peak')
    ax5.set_xlabel('Batch Size', fontsize=12)
    ax5.set_ylabel('GPU Utilization (%)', fontsize=12)
    ax5.set_title('GPU Utilization vs Batch Size', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_xscale('log', base=2)
    ax5.set_ylim([0, 105])
else:
    ax5.text(0.5, 0.5, 'GPU Monitoring Not Available', ha='center', va='center', 
             transform=ax5.transAxes, fontsize=14)
    ax5.set_title('GPU Utilization vs Batch Size', fontsize=14, fontweight='bold')

# Plot 6: GPU Memory Usage vs batch size
if GPU_AVAILABLE and any(results['gpu_memory_mean']):
    ax6.plot(results['batch_size'], results['gpu_memory_mean'], 'green', marker='o', linewidth=2, markersize=8)
    ax6.set_xlabel('Batch Size', fontsize=12)
    ax6.set_ylabel('GPU Memory Usage (%)', fontsize=12)
    ax6.set_title('GPU Memory Usage vs Batch Size', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_xscale('log', base=2)
    ax6.set_ylim([0, 105])
else:
    ax6.text(0.5, 0.5, 'GPU Monitoring Not Available', ha='center', va='center',
             transform=ax6.transAxes, fontsize=14)
    ax6.set_title('GPU Memory Usage vs Batch Size', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('batch_performance.png', dpi=150, bbox_inches='tight')
print("   ✓ Chart saved to: batch_performance.png")

# Summary table
print("\n6. Performance Summary Table")
print("=" * 100)
if GPU_AVAILABLE and any(results['gpu_util_mean']):
    print(f"{'Batch':<8} {'Total':<10} {'Per Sample':<12} {'Throughput':<14} {'Speedup':<10} {'GPU Util':<12} {'GPU Mem':<10}")
    print(f"{'Size':<8} {'(ms)':<10} {'(ms)':<12} {'(preds/s)':<14} {'':<10} {'(mean %)':<12} {'(%)':<10}")
else:
    print(f"{'Batch Size':<12} {'Total (ms)':<12} {'Per Sample (ms)':<18} {'Throughput (preds/s)':<20} {'Speedup':<10}")
print("-" * 100)
for i in range(len(results['batch_size'])):
    bs = results['batch_size'][i]
    total = results['total_time_ms'][i]
    per_sample = results['per_sample_ms'][i]
    throughput = results['throughput'][i]
    speedup = speedups[i]
    
    if GPU_AVAILABLE and any(results['gpu_util_mean']):
        gpu_util = results['gpu_util_mean'][i]
        gpu_mem = results['gpu_memory_mean'][i]
        print(f"{bs:<8} {total:<10.2f} {per_sample:<12.2f} {throughput:<14.1f} {speedup:<10.2f}x {gpu_util:<12.1f} {gpu_mem:<10.1f}")
    else:
        print(f"{bs:<12} {total:<12.2f} {per_sample:<18.2f} {throughput:<20.1f} {speedup:<10.2f}x")

print("\n" + "=" * 70)
print("Batch Processing Test Complete!")
print("=" * 70)

# Optimal batch size recommendation
optimal_idx = np.argmax(results['throughput'])
optimal_batch = results['batch_size'][optimal_idx]
optimal_throughput = results['throughput'][optimal_idx]

print(f"\n💡 Recommendation:")
print(f"   Optimal batch size: {optimal_batch}")
print(f"   Max throughput: {optimal_throughput:.1f} predictions/sec")
print(f"   Speedup: {speedups[optimal_idx]:.2f}x vs single inference")

