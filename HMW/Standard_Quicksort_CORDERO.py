import sys
import time
from typing import List, Tuple, Dict

class QuickSortMetrics:
    """Track quicksort performance metrics"""
    def __init__(self):
        self.comparisons = self.movements = self.max_depth = self.current_depth = self.peak_memory = 0
    
    def track_memory(self):
        mem = self.current_depth * sys.getsizeof(int) * 2
        if mem > self.peak_memory:
            self.peak_memory = mem

def standard_quicksort(arr: List[int]) -> Tuple[List[int], Dict[str, int]]:
    """
    Standard quicksort with instrumentation (Lomuto partition).
    Uses iterative approach with explicit stack to avoid recursion depth limits.
    Returns: (sorted_array, {comparisons, movements, peak_memory, max_depth})
    Time: O(n log n) avg, O(n²) worst | Space: O(log n) avg, O(n) worst
    """
    if not arr:
        return [], {"comparisons": 0, "movements": 0, "peak_memory": 0, "max_depth": 0}
    
    result, m = arr.copy(), QuickSortMetrics()
    
    def partition(low: int, high: int) -> int:
        pivot, i = result[high], low - 1
        for j in range(low, high):
            m.comparisons += 1
            if result[j] <= pivot:
                i += 1
                if i != j:
                    result[i], result[j] = result[j], result[i]
                    m.movements += 2
        if i + 1 != high:
            result[i + 1], result[high] = result[high], result[i + 1]
            m.movements += 2
        return i + 1
    
    # Iterative quicksort using explicit stack to avoid recursion limits
    stack = [(0, len(result) - 1)]
    
    while stack:
        low, high = stack.pop()
        
        if low < high:
            m.current_depth = len(stack) + 1
            m.max_depth = max(m.max_depth, m.current_depth)
            m.track_memory()
            
            pi = partition(low, high)
            
            # Push larger partition first for better space efficiency
            # This ensures stack depth stays O(log n) on average
            if pi - low > high - pi:
                stack.append((low, pi - 1))
                stack.append((pi + 1, high))
            else:
                stack.append((pi + 1, high))
                stack.append((low, pi - 1))
    
    return result, {"comparisons": m.comparisons, "movements": m.movements, 
                    "peak_memory": m.peak_memory, "max_depth": m.max_depth}

def run_tests(arr: List[int], runs: int = 5) -> Tuple[float, Dict[str, int]]:
    """Run multiple tests and return median time and metrics"""
    times, metrics_list = [], []
    for _ in range(runs):
        start = time.perf_counter()
        _, m = standard_quicksort(arr)
        times.append((time.perf_counter() - start) * 1000)
        metrics_list.append(m)
    
    times.sort()
    for key in metrics_list[0]:
        values = sorted([m[key] for m in metrics_list])
        metrics_list[0][key] = values[runs // 2]
    return times[runs // 2], metrics_list[0]

def fit_least_squares(x, y, transform):
    """Simple least squares: y = a*transform(x) + b"""
    import numpy as np
    n, x_t = len(x), transform(x)
    sx, sy, sxx, sxy = np.sum(x_t), np.sum(y), np.sum(x_t * x_t), np.sum(x_t * y)
    a = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    b = (sy - a * sx) / n
    return a, b

if __name__ == "__main__":
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Basic tests
    print("Standard Quicksort Performance Analysis")
    print("=" * 80 + "\n=== BASIC TEST CASES ===")
    
    for i, arr in enumerate([[64, 34, 25, 12, 22, 11, 90], [5, 2, 8, 1, 9], 
                             [1], [], [3, 3, 3, 3], [9, 8, 7, 6, 5, 4, 3, 2, 1]], 1):
        result, m = standard_quicksort(arr)
        print(f"\nTest {i}: {arr}\nSorted: {result}\nComparisons: {m['comparisons']}, "
              f"Movements: {m['movements']}, Max Depth: {m['max_depth']}, "
              f"Correct: {result == sorted(arr)}")
    
    # Scalability tests
    sizes = [10, 50, 100, 500, 1000, 2000, 5000, 10000]
    results = {'random': {'times': [], 'comps': []}, 
               'sorted': {'times': [], 'comps': []},
               'reverse': {'times': [], 'comps': []}}
    
    print("\n" + "=" * 80)
    print("=== SCALABILITY TESTS (Median of 5 runs) ===")
    print(f"{'Size':<10} {'Type':<10} {'Time (ms)':<12} {'Comparisons':<15} {'Movements':<12} {'Depth':<8}")
    print("-" * 80)
    
    for size in sizes:
        for case, gen in [('random', lambda: [random.randint(0, size*10) for _ in range(size)]),
                          ('sorted', lambda: list(range(size))),
                          ('reverse', lambda: list(range(size, 0, -1)))]:
            t, m = run_tests(gen())
            results[case]['times'].append(t)
            results[case]['comps'].append(m['comparisons'])
            print(f"{size:<10} {case:<10} {t:<12.4f} {m['comparisons']:<15} "
                  f"{m['movements']:<12} {m['max_depth']:<8}")
    
    # Detailed analysis
    print("\n" + "=" * 80 + "=== DETAILED ANALYSIS n=10000 ===")
    large_metrics = [standard_quicksort([random.randint(0, 100000) for _ in range(10000)])[1] 
                     for _ in range(5)]
    for key in ['comparisons', 'movements', 'max_depth', 'peak_memory']:
        vals = sorted([m[key] for m in large_metrics])
        median = vals[2]
        print(f"{key.title():20}: {median:8} (range: {vals[0]} - {vals[-1]})")
    
    # Curve fitting and plotting
    print("\n" + "=" * 80 + "=== CURVE FITTING & PLOTS ===")
    x = np.array(sizes, dtype=float)
    
    fits = {}
    for case in ['random', 'sorted', 'reverse']:
        y = np.array(results[case]['times'], dtype=float)
        transform = (lambda v: v * v) if case == 'reverse' else (lambda v: v * np.log2(v))
        a, b = fit_least_squares(x, y, transform)
        fits[case] = (a, b, transform)
        func_str = "n²" if case == 'reverse' else "n log₂(n)"
        print(f"{case.title():10}: T(n) = {a:.6f} * {func_str} + {b:.6f}")
    
    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    colors = {'random': 'blue', 'sorted': 'green', 'reverse': 'red'}
    markers = {'random': 'o', 'sorted': 's', 'reverse': '^'}
    labels = {'random': 'Random (Avg)', 'sorted': 'Sorted(Best)', 'reverse': 'Reverse(Worst)'}
    
    # Plot 1: Time vs Size
    for case in ['random', 'sorted', 'reverse']:
        axes[0,0].plot(sizes, results[case]['times'], f"{markers[case]}-", 
                       label=labels[case], linewidth=2, markersize=8, color=colors[case])
    axes[0,0].set_xlabel('Input Size (n)', fontweight='bold')
    axes[0,0].set_ylabel('Time (ms)', fontweight='bold')
    axes[0,0].set_title('Execution Time vs Input Size\n(Median of 5 runs)', fontweight='bold')
    axes[0,0].legend()
    axes[0,0].grid(alpha=0.3)
    
    # Plot 2: Curve Fitting
    x_smooth = np.linspace(min(sizes), max(sizes), 300)
    for case in ['random', 'reverse']:
        a, b, transform = fits[case]
        axes[0,1].plot(sizes, results[case]['times'], markers[case], 
                       label=f"{labels[case]}", markersize=8, color=colors[case])
        y_fit = a * transform(x_smooth) + b
        func = "n²" if case == 'reverse' else "n log n"
        axes[0,1].plot(x_smooth, y_fit, '--', label=f'O({func}) fit: {a:.4f}·{func}', 
                       linewidth=2, color=colors[case], alpha=0.7)
    axes[0,1].set_xlabel('Input Size (n)', fontweight='bold')
    axes[0,1].set_ylabel('Time (ms)', fontweight='bold')
    axes[0,1].set_title('Time Complexity Curve Fitting\n(Median of 5 runs)', fontweight='bold')
    axes[0,1].legend()
    axes[0,1].grid(alpha=0.3)
    
    # Plot 3: Comparisons
    for case in ['random', 'sorted', 'reverse']:
        axes[1,0].plot(sizes, results[case]['comps'], f"{markers[case]}-", 
                       label=labels[case], linewidth=2, markersize=8, color=colors[case])
    axes[1,0].set_xlabel('Input Size (n)', fontweight='bold')
    axes[1,0].set_ylabel('Comparisons', fontweight='bold')
    axes[1,0].set_title('Comparisons vs Input Size\n(Median of 5 runs)', fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3)
    
    # Plot 4: Normalized Complexity
    theoretical = {'random': x * np.log2(x), 'sorted': x * np.log2(x), 'reverse': x * x}
    for case in ['random', 'sorted', 'reverse']:
        norm = np.array(results[case]['comps']) / theoretical[case]
        axes[1,1].plot(sizes, norm, f"{markers[case]}-", 
                       label=f"{labels[case]} / theoretical", linewidth=2, markersize=8, color=colors[case])
    axes[1,1].axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    axes[1,1].set_xlabel('Input Size (n)', fontweight='bold')
    axes[1,1].set_ylabel('Normalized Comparisons', fontweight='bold')
    axes[1,1].set_title('Complexity Scaling Factor\n(Actual / Theoretical)', fontweight='bold')
    axes[1,1].legend()
    axes[1,1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quicksort_performance_analysis.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'quicksort_performance_analysis.png'")
    plt.show()
