
#!/usr/bin/env python3
"""VLM Gym 训练脚本 - MVP版本"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vlm_gym.environments.vision_qa_env import VisionQAEnvironment
from vlm_gym.agents.vlm_agent import VLMAgent
from vlm_gym.trainers.simple_trainer import SimpleTrainer

def main():
    """主训练函数"""
    print("🎪 欢迎使用 VLM Gym!")
    print("📝 Initial版本 - agent-environment视觉语言模型训练")
    
    # 配置
    config = {
        'model_path': 'Qwen/Qwen2.5-VL-7B-Instruct',
        'dataset_path': 'data/vision_r1_sample_dataset.json',
        'num_episodes': 5,  # 先试5个episodes
    }
    
    try:
        # 1. 创建环境
        print("\n🌍 创建环境...")
        env = VisionQAEnvironment(config['dataset_path'])
        
        # 2. 创建智能体
        print("\n🤖 创建智能体...")
        agent = VLMAgent(config['model_path'])
        
        # 3. 创建训练器
        print("\n🎓 创建训练器...")
        trainer = SimpleTrainer(agent, env, config)
        
        # 4. 开始训练
        print(f"\n🚀 开始训练...")
        results = trainer.train(config['num_episodes'])
        
        # 5. 总结
        total_rewards = [stats['reward'] for stats in results['training_log']]
        total_accuracy = [stats['accuracy'] for stats in results['training_log']]
        
        print(f"\n🎉 训练完成!")
        print(f"📊 总体统计:")
        print(f"   总episodes: {len(total_rewards)}")
        print(f"   平均奖励: {sum(total_rewards)/len(total_rewards):.3f}")
        print(f"   平均准确率: {sum(total_accuracy)/len(total_accuracy):.3f}")
        print(f"   最高奖励: {max(total_rewards):.3f}")
        
    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()