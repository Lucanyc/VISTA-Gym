
#!/usr/bin/env python3
"""VLM Gym training"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vlm_gym.environments.vision_qa_env import VisionQAEnvironment
from vlm_gym.agents.vlm_agent import VLMAgent
from vlm_gym.trainers.simple_trainer import SimpleTrainer

def main():
    """main function"""
    print("Welcome to use VLM-Gym!")
    print("Initial version- agent-environment VLM training")
    
    
    config = {
        'model_path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        'dataset_path': 'data/vision_r1_sample_dataset.json',
        'num_episodes': 5,  
    }
    
    try:
        # 1. create envrionment
        print("\n create envrionment...")
        env = VisionQAEnvironment(config['dataset_path'])
        
        # 2. create agents
        print("\n create agents...")
        agent = VLMAgent(config['model_path'])
        
        # 3. create trainner
        print("\n create trainner...")
        trainer = SimpleTrainer(agent, env, config)
        
        # 4. strat to train
        print(f"\n start training...")
        results = trainer.train(config['num_episodes'])
        
        # 5. summary
        total_rewards = [stats['reward'] for stats in results['training_log']]
        total_accuracy = [stats['accuracy'] for stats in results['training_log']]
        
        print(f"\n Training complete!")
        print(f"Statistics:")
        print(f"   Total episodes: {len(total_rewards)}")
        print(f"   Average reward: {sum(total_rewards)/len(total_rewards):.3f}")
        print(f"   Average accuracy: {sum(total_accuracy)/len(total_accuracy):.3f}")
        print(f"   Highest reward: {max(total_rewards):.3f}")
        
    except KeyboardInterrupt:
        print("\n training interrupted")
    except Exception as e:
        print(f"\n errors: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
