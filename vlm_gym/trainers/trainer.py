import time
from typing import Dict, Any

class SimpleTrainer:
    """initial version of trainer"""
    
    def __init__(self, agent, environment, config: Dict[str, Any] = None):
        self.agent = agent
        self.environment = environment
        self.config = config or {}
        self.training_log = []
    
    def train_episode(self) -> Dict[str, float]:
        """train a episode"""
        # reset the environmnet
        obs = self.environment.reset()
        
        print(f"\n question: {obs['question']}")
        print(f" Image: {obs['image_path']}")
        
        # agent generate answer
        start_time = time.time()
        action = self.agent.act(obs)
        inference_time = time.time() - start_time
        
        print(f"Answer: {action}")
        
        # feedback from environment
        next_obs, reward, done, info = self.environment.step(action)
        
        print(f"Reward: {reward:.3f}")
        print(f"Right answer: {info['ground_truth']}")
        print(f"Reference time: {inference_time:.2f}秒")
        
        # agent learning 
        experience = {
            'observation': obs,
            'action': action,
            'reward': reward,
            'info': info
        }
        self.agent.update(experience)
        
        return {
            'reward': reward,
            'accuracy': info['accuracy'],
            'inference_time': inference_time
        }
    
    def train(self, num_episodes: int = 10) -> Dict[str, Any]:
        """train multiple episodes"""
        print(f"Strat training {num_episodes} 个episodes...")
        
        for episode in range(num_episodes):
            print(f"\n{'='*50}")
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"{'='*50}")
            
            episode_stats = self.train_episode()
            self.training_log.append(episode_stats)
            
            # print statistics
            if (episode + 1) % 5 == 0:
                recent_rewards = [stats['reward'] for stats in self.training_log[-5:]]
                recent_accuracy = [stats['accuracy'] for stats in self.training_log[-5:]]
                
                print(f"\n latest five statistics:")
                print(f"   average reward: {sum(recent_rewards)/len(recent_rewards):.3f}")
                print(f"  average accuracy: {sum(recent_accuracy)/len(recent_accuracy):.3f}")
        
        return {'training_log': self.training_log}
