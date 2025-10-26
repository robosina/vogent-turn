import time
import numpy as np
from vogent_turn import TurnDetector
import soundfile as sf
import urllib.request
import threading
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

try:
    import pynvml
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: pynvml not installed. GPU monitoring disabled.")
    print("Install with: pip install nvidia-ml-py3")

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
                    # Get GPU utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                    
                    self.timestamps.append(time.time() - start_time)
                    self.gpu_utils.append(util.gpu)
                    self.gpu_memory.append(memory_info.used / memory_info.total * 100)
                except Exception as e:
                    print(f"GPU monitoring error: {e}")
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
    
    def plot(self, filename='gpu_utilization.png'):
        """Plot GPU utilization"""
        if not self.gpu_utils:
            print("No GPU data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # GPU Utilization
        ax1.plot(self.timestamps, self.gpu_utils, 'b-', linewidth=1, alpha=0.7)
        ax1.fill_between(self.timestamps, self.gpu_utils, alpha=0.3)
        ax1.set_ylabel('GPU Utilization (%)', fontsize=12)
        ax1.set_title('GPU Utilization Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 105])
        
        # Add statistics
        mean_util = np.mean(self.gpu_utils)
        max_util = np.max(self.gpu_utils)
        ax1.axhline(y=mean_util, color='r', linestyle='--', linewidth=1, 
                   label=f'Mean: {mean_util:.1f}%')
        ax1.legend(loc='upper right')
        
        # GPU Memory
        ax2.plot(self.timestamps, self.gpu_memory, 'g-', linewidth=1, alpha=0.7)
        ax2.fill_between(self.timestamps, self.gpu_memory, alpha=0.3, color='green')
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('GPU Memory (%)', fontsize=12)
        ax2.set_title('GPU Memory Usage Over Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 105])
        
        # Add statistics
        mean_mem = np.mean(self.gpu_memory)
        ax2.axhline(y=mean_mem, color='r', linestyle='--', linewidth=1,
                   label=f'Mean: {mean_mem:.1f}%')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n   GPU utilization chart saved to: {filename}")
    
    def __del__(self):
        """Cleanup"""
        if GPU_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


print("=" * 70)
print("Vogent Turn Benchmark")
print("=" * 70)

# Initialize GPU monitor
gpu_monitor = GPUMonitor()

# Download sample audio
print("\n1. Downloading sample audio...")
audio_url = "https://storage.googleapis.com/voturn-sample-recordings/incomplete_number_sample.wav"
urllib.request.urlretrieve(audio_url, "sample.wav")
audio, sr = sf.read("sample.wav")
print(f"   Audio loaded: {len(audio)} samples at {sr}Hz ({len(audio)/sr:.2f} seconds)")

# Initialize detector
print("\n2. Initializing TurnDetector...")
print("   - With torch.compile and warmup (optimized)")
start_init = time.time()
detector = TurnDetector(compile_model=True, warmup=True)
init_time = time.time() - start_init
print(f"   Initialization time: {init_time:.2f}s")

# Prepare test data
prev_line = "What is your phone number"
curr_line = "My number is 804"

# Warmup runs (already done in TurnDetector init, but let's do a few more)
print("\n3. Warmup runs (3 iterations)...")
for i in range(3):
    _ = detector.predict(audio, prev_line=prev_line, curr_line=curr_line, sample_rate=sr)
print("   Warmup complete")

# Benchmark single inference
print("\n4. Benchmarking single inference...")
num_iterations = 100
times = []

# Start GPU monitoring
gpu_monitor.start()

for i in range(num_iterations):
    start = time.time()
    result = detector.predict(
        audio,
        prev_line=prev_line,
        curr_line=curr_line,
        sample_rate=sr,
        return_probs=True
    )
    elapsed = (time.time() - start) * 1000  # Convert to milliseconds
    times.append(elapsed)

# Stop GPU monitoring
gpu_monitor.stop()

# Calculate statistics
times = np.array(times)
mean_time = np.mean(times)
std_time = np.std(times)
min_time = np.min(times)
max_time = np.max(times)
p50_time = np.percentile(times, 50)
p95_time = np.percentile(times, 95)
p99_time = np.percentile(times, 99)

print(f"\n   Results over {num_iterations} iterations:")
print(f"   - Mean:   {mean_time:.2f} ms (± {std_time:.2f} ms)")
print(f"   - Median: {p50_time:.2f} ms")
print(f"   - Min:    {min_time:.2f} ms")
print(f"   - Max:    {max_time:.2f} ms")
print(f"   - P95:    {p95_time:.2f} ms")
print(f"   - P99:    {p99_time:.2f} ms")
print(f"\n   Throughput: {1000/mean_time:.1f} predictions/second")

# Display GPU stats
gpu_stats = gpu_monitor.get_stats()
if gpu_stats:
    print(f"\n   GPU Statistics:")
    print(f"   - Mean Utilization: {gpu_stats['mean_util']:.1f}%")
    print(f"   - Peak Utilization: {gpu_stats['max_util']:.1f}%")
    print(f"   - Mean Memory: {gpu_stats['mean_memory']:.1f}%")
    print(f"   - Peak Memory: {gpu_stats['max_memory']:.1f}%")

# Show prediction result
print("\n5. Sample prediction result:")
print(f"   - Turn complete: {result['is_endpoint']}")
print(f"   - Endpoint probability: {result['prob_endpoint']:.1%}")
print(f"   - Continue probability: {result['prob_continue']:.1%}")

# Batch inference benchmark
print("\n6. Benchmarking batch inference (batch_size=8)...")
batch_size = 8
audio_batch = [audio] * batch_size
context_batch = [{'prev_line': prev_line, 'curr_line': curr_line}] * batch_size

batch_times = []
for i in range(20):  # Fewer iterations for batch
    start = time.time()
    results = detector.predict_batch(
        audio_batch,
        context_batch=context_batch,
        sample_rate=sr,
        return_probs=True
    )
    elapsed = (time.time() - start) * 1000
    batch_times.append(elapsed)

batch_times = np.array(batch_times)
mean_batch_time = np.mean(batch_times)
per_sample_time = mean_batch_time / batch_size

print(f"\n   Results over 20 iterations:")
print(f"   - Mean batch time: {mean_batch_time:.2f} ms")
print(f"   - Time per sample: {per_sample_time:.2f} ms")
print(f"   - Throughput: {1000/per_sample_time:.1f} predictions/second")
print(f"   - Speedup vs single: {mean_time/per_sample_time:.2f}x")

print("\n" + "=" * 70)
print("Generating GPU utilization charts...")
print("=" * 70)

# Generate GPU utilization chart
if GPU_AVAILABLE and gpu_stats:
    gpu_monitor.plot('gpu_utilization.png')
    print("\n✓ Chart saved successfully!")
else:
    print("\n⚠ GPU monitoring not available. No chart generated.")

print("\n" + "=" * 70)
print("Benchmark complete!")
print("=" * 70)

