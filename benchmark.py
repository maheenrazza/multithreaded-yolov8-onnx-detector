#!/usr/bin/env python3

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Thread
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil


@dataclass
class SystemInfo:
    cpu_count_physical: int
    cpu_count_logical: int
    cpu_freq_mhz: float
    total_memory_gb: float
    platform: str
    python_version: str


@dataclass
class BenchmarkMetrics:
    gflops: float
    memory_bandwidth_gbps: float
    benchmark_time_ms: float


@dataclass
class StudentResult:
    exit_code: int
    elapsed_sec: float
    peak_memory_mb: float
    avg_cpu_percent: float
    stdout_lines: List[str]
    stderr_lines: List[str]


@dataclass
class PerformanceScore:
    compute_efficiency: float
    memory_efficiency: float
    time_efficiency: float
    overall_score: float


class SystemProfiler:
    @staticmethod
    def get_system_info() -> SystemInfo:
        try:
            cpu_freq = psutil.cpu_freq()
            freq = cpu_freq.current if cpu_freq else 0.0
        except:
            freq = 0.0

        try:
            memory = psutil.virtual_memory()
            total_mem_gb = memory.total / (1024**3)
        except:
            total_mem_gb = 0.0

        return SystemInfo(
            cpu_count_physical=psutil.cpu_count(logical=False) or 1,
            cpu_count_logical=psutil.cpu_count(logical=True) or 1,
            cpu_freq_mhz=freq,
            total_memory_gb=total_mem_gb,
            platform=platform.platform(),
            python_version=platform.python_version()
        )


class HardwareBenchmark:
    def __init__(self, matrix_size: int = 512, memory_mb: int = 200):
        self.matrix_size = matrix_size
        self.memory_mb = memory_mb

    def measure_compute_performance(self, iterations: int = 3) -> Tuple[float, float]:
        np.random.seed(42)
        matrix_a = np.random.rand(self.matrix_size, self.matrix_size).astype(np.float32)
        matrix_b = np.random.rand(self.matrix_size, self.matrix_size).astype(np.float32)
        
        # Warmup
        _ = np.dot(matrix_a, matrix_b)
        
        start_time = time.perf_counter()
        for _ in range(iterations):
            result = np.dot(matrix_a, matrix_b)
            del result
        
        elapsed = time.perf_counter() - start_time
        total_flops = 2.0 * (self.matrix_size ** 3) * iterations
        gflops = (total_flops / elapsed) * 1e-9 if elapsed > 0 else 0.0
        
        return gflops, elapsed * 1000

    def measure_memory_bandwidth(self, iterations: int = 5) -> Tuple[float, float]:
        size_bytes = self.memory_mb * 1024 * 1024
        source = np.random.bytes(size_bytes)
        source_array = np.frombuffer(source, dtype=np.uint8)
        dest_array = np.empty_like(source_array)
        
        start_time = time.perf_counter()
        total_bytes = 0
        
        for _ in range(iterations):
            dest_array[:] = source_array[:]
            total_bytes += size_bytes
        
        elapsed = time.perf_counter() - start_time
        gbps = (total_bytes / (1024**3)) / elapsed if elapsed > 0 else 0.0
        
        return gbps, elapsed * 1000

    def run_benchmark(self) -> BenchmarkMetrics:
        gflops, compute_time = self.measure_compute_performance()
        bandwidth, memory_time = self.measure_memory_bandwidth()
        total_time = compute_time + memory_time
        
        return BenchmarkMetrics(gflops, bandwidth, total_time)


class StudentCodeRunner:
    def __init__(self, sample_interval: float = 0.1, timeout: Optional[float] = None):
        self.sample_interval = sample_interval
        self.timeout = timeout

    def run_student_code(self, binary_path: str, args: List[str]) -> StudentResult:
        if not Path(binary_path).is_file():
            raise FileNotFoundError(f"Binary not found: {binary_path}")

        cmd = [binary_path] + args
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            psutil_proc = psutil.Process(process.pid)
            
            start_time = time.perf_counter()
            peak_memory = 0
            cpu_samples = []
            
            stdout_lines = []
            stderr_lines = []
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                stdout_future = executor.submit(self._read_output, process.stdout)
                stderr_future = executor.submit(self._read_output, process.stderr)
                
                while process.poll() is None:
                    elapsed = time.perf_counter() - start_time
                    if self.timeout and elapsed > self.timeout:
                        process.kill()
                        break
                    
                    try:
                        memory_info = psutil_proc.memory_info()
                        peak_memory = max(peak_memory, memory_info.rss)
                        
                        cpu_percent = psutil_proc.cpu_percent()
                        if cpu_percent > 0:
                            cpu_samples.append(cpu_percent)
                    
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        break
                    
                    time.sleep(self.sample_interval)
                
                stdout_lines = stdout_future.result()
                stderr_lines = stderr_future.result()
            
            elapsed_time = time.perf_counter() - start_time
            exit_code = process.returncode or 0
            avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
            
            return StudentResult(
                exit_code=exit_code,
                elapsed_sec=elapsed_time,
                peak_memory_mb=peak_memory / (1024 * 1024),
                avg_cpu_percent=avg_cpu,
                stdout_lines=stdout_lines[:20],
                stderr_lines=stderr_lines[:10]
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to run student code: {e}")

    def _read_output(self, stream) -> List[str]:
        try:
            return [line.strip() for line in stream if line.strip()]
        except:
            return []


class PerformanceEvaluator:
    def __init__(self, baseline_gflops: float = 50.0):
        self.baseline_gflops = baseline_gflops

    def calculate_scores(
        self,
        benchmark: BenchmarkMetrics,
        student_result: StudentResult,
        system_info: SystemInfo
    ) -> PerformanceScore:
        
        # Normalize by system capabilities
        compute_efficiency = self._calculate_compute_efficiency(
            benchmark.gflops, student_result.elapsed_sec, student_result.avg_cpu_percent
        )
        
        memory_efficiency = self._calculate_memory_efficiency(
            benchmark.memory_bandwidth_gbps, student_result.peak_memory_mb, system_info.total_memory_gb
        )
        
        time_efficiency = self._calculate_time_efficiency(
            student_result.elapsed_sec, benchmark.gflops
        )
        
        overall_score = (
            0.4 * compute_efficiency +
            0.3 * memory_efficiency +
            0.3 * time_efficiency
        )
        
        return PerformanceScore(
            compute_efficiency=round(compute_efficiency, 4),
            memory_efficiency=round(memory_efficiency, 4),
            time_efficiency=round(time_efficiency, 4),
            overall_score=round(overall_score, 4)
        )

    def _calculate_compute_efficiency(self, sys_gflops: float, elapsed: float, cpu_usage: float) -> float:
        if elapsed <= 0 or sys_gflops <= 0:
            return 0.0
        
        # Higher GFLOPS and lower time = better efficiency
        base_score = (sys_gflops / elapsed) / 100.0
        
        # Bonus for efficient CPU usage (not maxing out unnecessarily)
        cpu_efficiency = 1.0 if cpu_usage == 0 else min(1.0, (100.0 - abs(cpu_usage - 80.0)) / 100.0)
        
        return min(1.0, base_score * cpu_efficiency)

    def _calculate_memory_efficiency(self, sys_bandwidth: float, peak_mem_mb: float, total_mem_gb: float) -> float:
        if total_mem_gb <= 0 or peak_mem_mb <= 0:
            return 1.0
        
        memory_usage_ratio = peak_mem_mb / (total_mem_gb * 1024)
        
        # Penalize excessive memory usage
        if memory_usage_ratio > 0.8:
            efficiency = 0.2
        elif memory_usage_ratio > 0.5:
            efficiency = 0.6
        elif memory_usage_ratio > 0.2:
            efficiency = 0.9
        else:
            efficiency = 1.0
        
        # Bonus for systems with good memory bandwidth
        bandwidth_factor = min(1.2, sys_bandwidth / 20.0)
        
        return min(1.0, efficiency * bandwidth_factor)

    def _calculate_time_efficiency(self, elapsed: float, sys_gflops: float) -> float:
        if elapsed <= 0:
            return 0.0
        
        # Normalize time based on system compute capability
        expected_time = 10.0 / (sys_gflops / self.baseline_gflops)
        efficiency = expected_time / elapsed if elapsed > expected_time else 1.0
        
        return min(1.0, efficiency)


class BenchmarkRunner:
    def __init__(self, config: Dict):
        self.config = config
        self.system_profiler = SystemProfiler()
        self.hw_benchmark = HardwareBenchmark(
            config.get('matrix_size', 512),
            config.get('memory_mb', 200)
        )
        self.code_runner = StudentCodeRunner(
            config.get('sample_interval', 0.1),
            config.get('timeout')
        )
        self.evaluator = PerformanceEvaluator(config.get('baseline_gflops', 50.0))

    def run_full_benchmark(self, binary_path: str, args: List[str]) -> Dict:
        print("Gathering system information...")
        system_info = self.system_profiler.get_system_info()
        
        print("Running hardware benchmark...")
        benchmark_metrics = self.hw_benchmark.run_benchmark()
        
        print("Running student code...")
        student_result = self.code_runner.run_student_code(binary_path, args)
        
        print("Calculating performance scores...")
        scores = self.evaluator.calculate_scores(benchmark_metrics, student_result, system_info)
        
        results = {
            'timestamp': time.time(),
            'system_info': asdict(system_info),
            'benchmark_metrics': asdict(benchmark_metrics),
            'student_result': asdict(student_result),
            'performance_scores': asdict(scores),
            'config': self.config
        }
        
        return results

    def save_results(self, results: Dict, output_file: str = 'benchmark_results.json'):
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Student Code Performance Benchmark")
    parser.add_argument("--binary", required=True, help="Path to student binary")
    parser.add_argument("--output", default="benchmark_results.json", help="Output JSON file")
    parser.add_argument("--matrix-size", type=int, default=512, help="Matrix size for compute benchmark")
    parser.add_argument("--memory-mb", type=int, default=200, help="Memory size for bandwidth test (MB)")
    parser.add_argument("--timeout", type=float, help="Timeout for student code (seconds)")
    parser.add_argument("--baseline-gflops", type=float, default=50.0, help="Baseline GFLOPS for normalization")
    parser.add_argument("args", nargs="*", help="Arguments to pass to student binary")
    
    parsed_args = parser.parse_args()
    
    config = {
        'matrix_size': parsed_args.matrix_size,
        'memory_mb': parsed_args.memory_mb,
        'timeout': parsed_args.timeout,
        'baseline_gflops': parsed_args.baseline_gflops,
        'sample_interval': 0.1
    }
    
    runner = BenchmarkRunner(config)
    
    try:
        results = runner.run_full_benchmark(parsed_args.binary, parsed_args.args)
        runner.save_results(results, parsed_args.output)
        
        print("\n" + "="*50)
        print("BENCHMARK SUMMARY")
        print("="*50)
        
        scores = results['performance_scores']
        system = results['system_info']
        student = results['student_result']
        
        print(f"System: {system['platform']}")
        print(f"CPU: {system['cpu_count_physical']}C/{system['cpu_count_logical']}T @ {system['cpu_freq_mhz']:.0f}MHz")
        print(f"Memory: {system['total_memory_gb']:.1f}GB")
        print(f"Exit Code: {student['exit_code']}")
        print(f"Execution Time: {student['elapsed_sec']:.2f}s")
        print(f"Peak Memory: {student['peak_memory_mb']:.1f}MB")
        print(f"Avg CPU Usage: {student['avg_cpu_percent']:.1f}%")
        print("\nPerformance Scores:")
        print(f"  Compute Efficiency: {scores['compute_efficiency']:.4f}")
        print(f"  Memory Efficiency:  {scores['memory_efficiency']:.4f}")
        print(f"  Time Efficiency:    {scores['time_efficiency']:.4f}")
        print(f"  Overall Score:      {scores['overall_score']:.4f}")
        
    except Exception as e:
        print(f"Benchmark failed: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())