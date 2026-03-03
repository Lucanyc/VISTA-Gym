# test_df_with_geometry3k_complete.py
import sys
sys.path.append('/data/wang/meng/GYM-Work/vlm_gym-tool-usage-geometry')

from vlm_gym.environments import VisionQAEnv
from vlm_gym.environments.task.geometry3k import Geometry3KTask
from data_adapters.geometry3k_adapter import Geometry3KAdapter

print("=== 测试 DiagramFormalizer 与 Geometry3K (完整版) ===\n")

# 设置路径
dataset_path = "/workspace/geometry/data/geometry3k"
try:
    # 1. 创建 adapter
    print("创建 Geometry3K Adapter...")
    adapter = Geometry3KAdapter(
        data_root=dataset_path,
        split="train",  # 或 "val", "test"
        validate_images=False  # 暂时关闭图像验证，加快速度
    )
    
    # 获取一些任务ID
    task_ids = adapter.get_task_ids(limit=5)
    print(f"找到 {len(task_ids)} 个任务")
    
    if not task_ids:
        print("警告：没有找到任务，可能是数据路径问题")
        # 尝试使用默认路径
        adapter = Geometry3KAdapter()
        task_ids = adapter.get_task_ids(limit=5)
    
    if task_ids:
        print(f"示例任务ID: {task_ids[:3]}")
        
        # 2. 创建环境
        print("\n创建 VisionQAEnv...")
        env = VisionQAEnv(
            dataset_path=dataset_path,
            enable_diagram_formalizer=True,
            task_kwargs={"adapter": adapter}  # 传递 adapter 给任务
        )
        
        # 手动设置 task_entrypoint
        env.task_entrypoint = Geometry3KTask
        
        # 3. 测试第一个任务
        print(f"\n测试任务: {task_ids[0]}")
        obs, info = env.reset(task_id=task_ids[0])
        
        print(f"\n任务信息:")
        print(f"- 问题: {obs.get('question', 'N/A')[:100]}...")
        print(f"- 有图像: {obs.get('has_image', False)}")
        print(f"- 可用工具: {obs.get('available_tools', [])}")
        
        # 4. 测试 DiagramFormalizer
        if 'diagram_formalizer' in obs.get('available_tools', []):
            print("\n使用 DiagramFormalizer 求解...")
            
            tool_call = f"""<tool_call>
{{
    "tool": "diagram_formalizer",
    "parameters": {{
        "task": "solve",
        "problem": "{obs['question']}"
    }}
}}
</tool_call>"""
            
            obs2, reward, done, truncated, info2 = env.step(tool_call)
            
            print(f"\n工具执行结果:")
            print(f"- 奖励: {reward}")
            print(f"- 完成: {done}")
            
            if 'tool_feedback' in obs2:
                feedback = obs2['tool_feedback']
                print(f"- 任务类型: {feedback.get('task_type')}")
                
                # 显示解答
                if feedback.get('solution'):
                    solution = feedback.get('solution', '')
                    print(f"\n工具解答:")
                    print(f"{solution[:400]}{'...' if len(solution) > 400 else ''}")
                
                # 显示形式化输出
                if feedback.get('formalized_output'):
                    formal = feedback.get('formalized_output', '')
                    print(f"\n形式化输出:")
                    print(f"{formal[:300]}{'...' if len(formal) > 300 else ''}")
        
        # 5. 测试更多任务类型
        print("\n\n测试不同的任务类型...")
        task_types = ["formalize", "analyze", "extract_constraints"]
        
        for task_type in task_types[:2]:  # 测试前两个
            print(f"\n任务类型: {task_type}")
            
            tool_call = f"""<tool_call>
{{
    "tool": "diagram_formalizer",
    "parameters": {{
        "task": "{task_type}",
        "problem": "{obs['question']}"
    }}
}}
</tool_call>"""
            
            obs3, reward3, done3, _, info3 = env.step(tool_call)
            print(f"- 奖励: {reward3}")
            
            if 'tool_feedback' in obs3:
                feedback = obs3['tool_feedback']
                if task_type == "extract_constraints" and feedback.get('constraints'):
                    print(f"- 提取到 {len(feedback.get('constraints', []))} 个约束")
                elif feedback.get('formalized_output'):
                    output = feedback.get('formalized_output', '')
                    print(f"- 输出: {output[:150]}...")
        
        env.close()
        print("\n✅ 测试成功完成!")
        
    else:
        print("❌ 没有找到任何任务，请检查数据集路径")
        
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()