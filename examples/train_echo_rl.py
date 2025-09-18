"""
EchoRL Training Example

This example demonstrates how to train EchoRL on various tasks including:
- ALFWorld: Text-world control tasks
- WebShop: Web-based shopping agent tasks
- CRUXEval: Code repair and debugging tasks
- ARC: Abstract reasoning tasks
- MiniGrid: Grid-world planning tasks

Usage:
    python examples/train_echo_rl.py --task alfworld --backbone gpt-4o --timesteps 100000
"""

import argparse
import asyncio
import logging
import torch
import os
from pathlib import Path

# Add the project root to Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from echo_rl.training.trainer import EchoRLTrainer, TrainingConfig
from echo_rl.core.latent_planning import PlanningConfig
from echo_rl.core.async_execution import ExecutionConfig
from echo_rl.core.prioritized_replay import ReplayConfig
from echo_rl.core.ppo_learner import PPOConfig
from echo_rl.utils.monitoring import PerformanceMonitor, MetricsLogger

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('echo_rl_training.log')
        ]
    )

def create_training_config(args) -> TrainingConfig:
    """Create training configuration from command line arguments"""
    
    # Component configurations
    planning_config = PlanningConfig(
        embedding_dim=512,
        state_window_size=8,
        kl_weight=0.1,
        learning_rate=3e-4
    )
    
    execution_config = ExecutionConfig(
        max_concurrent_rollouts=128,
        max_cache_size=10000,
        timeout=30.0
    )
    
    replay_config = ReplayConfig(
        hot_buffer_size=1000000,
        cold_buffer_size=10000000,
        age_threshold=1000,
        temperature=1.0
    )
    
    ppo_config = PPOConfig(
        learning_rate=3e-4,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        kl_coef=0.1,
        gae_lambda=0.95,
        gamma=0.99
    )
    
    # Environment configuration
    env_config = {}
    if args.task == "alfworld":
        env_config = {
            "task_type": "pick_and_place",
            "max_objects": 10,
            "room_size": 5
        }
    elif args.task == "webshop":
        env_config = {
            "website_type": "electronics",
            "max_search_results": 20,
            "budget_limit": 1000.0
        }
    elif args.task == "cruxeval":
        env_config = {
            "language": "python",
            "max_code_length": 1000,
            "max_test_cases": 10
        }
    elif args.task == "arc":
        env_config = {
            "grid_size": 10,
            "max_colors": 10,
            "task_type": "pattern_completion"
        }
    elif args.task == "minigrid":
        env_config = {
            "grid_size": 8,
            "num_objects": 3,
            "task_type": "key_door"
        }
    
    # Main training configuration
    config = TrainingConfig(
        env_name=args.task,
        env_config=env_config,
        total_timesteps=args.timesteps,
        learning_starts=10000,
        train_frequency=4,
        evaluation_frequency=10000,
        save_frequency=50000,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=args.seed,
        num_actors=args.num_actors,
        num_learners=args.num_learners,
        batch_size=args.batch_size,
        planning_config=planning_config,
        execution_config=execution_config,
        replay_config=replay_config,
        ppo_config=ppo_config,
        checkpoint_dir=f"./checkpoints/{args.task}_{args.backbone}",
        results_dir=f"./results/{args.task}_{args.backbone}",
        tensorboard_log=f"./logs/tensorboard/{args.task}_{args.backbone}"
    )
    
    return config

async def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train EchoRL on various tasks")
    
    # Task and model arguments
    parser.add_argument("--task", type=str, default="alfworld",
                       choices=["alfworld", "webshop", "cruxeval", "arc", "minigrid"],
                       help="Task to train on")
    parser.add_argument("--backbone", type=str, default="gpt-4o",
                       help="Model backbone to use")
    
    # Training arguments
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Total training timesteps")
    parser.add_argument("--num-actors", type=int, default=128,
                       help="Number of actor processes")
    parser.add_argument("--num-learners", type=int, default=2,
                       help="Number of learner processes")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Training batch size")
    
    # System arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device to use for training")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting EchoRL training")
    logger.info(f"Task: {args.task}")
    logger.info(f"Backbone: {args.backbone}")
    logger.info(f"Timesteps: {args.timesteps}")
    logger.info(f"Device: {args.device}")
    
    # Create training configuration
    config = create_training_config(args)
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.tensorboard_log, exist_ok=True)
    
    # Initialize performance monitoring
    performance_monitor = PerformanceMonitor()
    metrics_logger = MetricsLogger(
        log_file=os.path.join(config.results_dir, "training_metrics.log")
    )
    
    # Start monitoring
    performance_monitor.start_monitoring()
    
    try:
        # Create trainer
        trainer = EchoRLTrainer(config)
        
        # Run training
        logger.info("Starting training loop...")
        metrics = await trainer.train()
        
        # Log final metrics
        logger.info("Training completed successfully!")
        logger.info(f"Final success rate: {metrics.evaluation_results.get('success_rate', 0.0):.3f}")
        logger.info(f"Final avg reward: {metrics.evaluation_results.get('avg_reward', 0.0):.3f}")
        logger.info(f"Total training time: {metrics.training_time:.2f} seconds")
        
        # Export performance data
        performance_monitor.export_performance_data(
            os.path.join(config.results_dir, "performance_report.json")
        )
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        # Cleanup
        performance_monitor.stop_monitoring()
        if 'trainer' in locals():
            trainer.close()
        
        logger.info("Training session ended")

if __name__ == "__main__":
    asyncio.run(main())
