"""
EchoRL Benchmarking Example

This example demonstrates how to run comprehensive benchmarks comparing EchoRL
against baseline methods across multiple tasks and backbones.

Usage:
    python examples/benchmark_echo_rl.py --tasks alfworld webshop --backbones gpt-4o claude-3.5-sonnet
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path

# Add the project root to Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from echo_rl.evaluation.benchmark import EchoRLBenchmark, BenchmarkConfig

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('echo_rl_benchmark.log')
        ]
    )

def create_benchmark_config(args) -> BenchmarkConfig:
    """Create benchmark configuration from command line arguments"""
    
    # EchoRL specific configurations
    echo_rl_configs = {
        "total_timesteps": args.timesteps,
        "num_actors": args.num_actors,
        "num_learners": args.num_learners,
        "batch_size": args.batch_size,
        "device": "cuda" if args.device == "auto" else args.device
    }
    
    # Task-specific configurations
    task_configs = {}
    
    if "alfworld" in args.tasks:
        task_configs["alfworld"] = {
            "task_type": "pick_and_place",
            "max_objects": 10,
            "room_size": 5
        }
    
    if "webshop" in args.tasks:
        task_configs["webshop"] = {
            "website_type": "electronics",
            "max_search_results": 20,
            "budget_limit": 1000.0
        }
    
    if "cruxeval" in args.tasks:
        task_configs["cruxeval"] = {
            "language": "python",
            "max_code_length": 1000,
            "max_test_cases": 10
        }
    
    if "arc" in args.tasks:
        task_configs["arc"] = {
            "grid_size": 10,
            "max_colors": 10,
            "task_type": "pattern_completion"
        }
    
    if "minigrid" in args.tasks:
        task_configs["minigrid"] = {
            "grid_size": 8,
            "num_objects": 3,
            "task_type": "key_door"
        }
    
    config = BenchmarkConfig(
        tasks=args.tasks,
        task_configs=task_configs,
        backbones=args.backbones,
        baselines=args.baselines,
        num_seeds=args.num_seeds,
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        evaluation_timeout=args.timeout,
        echo_rl_configs=echo_rl_configs,
        output_dir=args.output_dir,
        save_detailed_results=True,
        generate_plots=True,
        generate_report=True
    )
    
    return config

async def main():
    """Main benchmarking function"""
    parser = argparse.ArgumentParser(description="Benchmark EchoRL against baseline methods")
    
    # Task and model arguments
    parser.add_argument("--tasks", nargs="+", default=["alfworld"],
                       choices=["alfworld", "webshop", "cruxeval", "arc", "minigrid"],
                       help="Tasks to evaluate")
    parser.add_argument("--backbones", nargs="+", default=["gpt-4o"],
                       help="Model backbones to evaluate")
    parser.add_argument("--baselines", nargs="+", default=["react", "tot", "ppo-rlhf"],
                       help="Baseline methods to compare against")
    
    # Evaluation arguments
    parser.add_argument("--num-seeds", type=int, default=10,
                       help="Number of random seeds for evaluation")
    parser.add_argument("--num-episodes", type=int, default=100,
                       help="Number of episodes per evaluation")
    parser.add_argument("--max-steps", type=int, default=1000,
                       help="Maximum steps per episode")
    parser.add_argument("--timeout", type=float, default=300.0,
                       help="Evaluation timeout in seconds")
    
    # EchoRL training arguments
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Training timesteps for EchoRL")
    parser.add_argument("--num-actors", type=int, default=128,
                       help="Number of actor processes")
    parser.add_argument("--num-learners", type=int, default=2,
                       help="Number of learner processes")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Training batch size")
    
    # System arguments
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device to use for training")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting EchoRL benchmark evaluation")
    logger.info(f"Tasks: {args.tasks}")
    logger.info(f"Backbones: {args.backbones}")
    logger.info(f"Baselines: {args.baselines}")
    logger.info(f"Number of seeds: {args.num_seeds}")
    logger.info(f"Number of episodes: {args.num_episodes}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create benchmark configuration
    config = create_benchmark_config(args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Create benchmark
        benchmark = EchoRLBenchmark(config)
        
        # Run benchmark
        logger.info("Starting benchmark evaluation...")
        results = await benchmark.run_benchmark()
        
        # Log summary results
        logger.info("Benchmark evaluation completed successfully!")
        logger.info(f"Total evaluation time: {results.total_evaluation_time:.2f} seconds")
        
        # Print summary table
        logger.info("\n=== Summary Results ===")
        logger.info("Task | Method | Success Rate | Avg Reward | Tokens/sec")
        logger.info("-----|--------|--------------|------------|------------")
        
        for task in args.tasks:
            for method in args.baselines + ["echo_rl"]:
                if method in results.success_rates.get(task, {}):
                    sr = results.success_rates[task][method]
                    ar = results.avg_rewards[task][method]
                    tps = results.tokens_per_second[task][method]
                    logger.info(f"{task:8} | {method:8} | {sr:11.3f} | {ar:9.3f} | {tps:9.1f}")
        
        # Print EchoRL improvements
        logger.info("\n=== EchoRL Improvements ===")
        for task in args.tasks:
            echo_rl_sr = results.success_rates[task].get("echo_rl", 0.0)
            logger.info(f"\n{task.title()}:")
            
            for baseline in args.baselines:
                if baseline in results.success_rates[task]:
                    baseline_sr = results.success_rates[task][baseline]
                    improvement = ((echo_rl_sr - baseline_sr) / baseline_sr) * 100 if baseline_sr > 0 else 0
                    logger.info(f"  {baseline.upper()}: {baseline_sr:.3f} → EchoRL: {echo_rl_sr:.3f} "
                               f"({improvement:+.1f}% improvement)")
        
        logger.info(f"\nDetailed results saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        logger.info("Benchmark evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark evaluation failed with error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
