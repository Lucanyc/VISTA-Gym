import time
from typing import Dict, Any

class SimpleTrainer:
    """简单训练器 - MVP版本"""
    
    def __init__(self, agent, environment, config: Dict[str, Any] = None):
        self.agent = agent
        self.environment = environment
        self.config = config or {}
        self.training_log = []
    
    def train_episode(self) -> Dict[str, float]:
        """训练一个episode"""
        # 重置环境
        obs = self.environment.reset()
        
        print(f"\n🎯 问题: {obs['question']}")
        print(f"📁 图像: {obs['image_path']}")
        
        # 智能体生成回答
        start_time = time.time()
        action = self.agent.act(obs)
        inference_time = time.time() - start_time
        
        print(f"🤖 回答: {action}")
        
        # 环境给出反馈
        next_obs, reward, done, info = self.environment.step(action)
        
        print(f"🏆 奖励: {reward:.3f}")
        print(f"✅ 正确答案: {info['ground_truth']}")
        print(f"⏱️ 推理时间: {inference_time:.2f}秒")
        
        # 智能体学习
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
        """训练多个episodes"""
        print(f"🚀 开始训练 {num_episodes} 个episodes...")
        
        for episode in range(num_episodes):
            print(f"\n{'='*50}")
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"{'='*50}")
            
            episode_stats = self.train_episode()
            self.training_log.append(episode_stats)
            
            # 打印统计信息
            if (episode + 1) % 5 == 0:
                recent_rewards = [stats['reward'] for stats in self.training_log[-5:]]
                recent_accuracy = [stats['accuracy'] for stats in self.training_log[-5:]]
                
                print(f"\n📊 近5次统计:")
                print(f"   平均奖励: {sum(recent_rewards)/len(recent_rewards):.3f}")
                print(f"   平均准确率: {sum(recent_accuracy)/len(recent_accuracy):.3f}")
        
        return {'training_log': self.training_log}
