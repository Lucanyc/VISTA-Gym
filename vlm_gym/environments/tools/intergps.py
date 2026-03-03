# vlm_gym/environments/tools/intergps.py

import os
import sys
import json
import re
import time
import math
from typing import Dict, Any, List, Tuple, Optional
import logging
import random
from func_timeout import func_timeout, FunctionTimedOut

from vlm_gym.environments.tools.base import ToolBase

# 设置日志
logger = logging.getLogger(__name__)

# Inter-GPS路径设置 - 自动检测
possible_bases = [
    "/workspace/mathvista/InterGPS",
    "/data/wang/meng/GYM-Work/vlm_gym-tool-usage-mathvista/InterGPS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "InterGPS")
]

INTERGPS_BASE = None
for base in possible_bases:
    if os.path.exists(base):
        INTERGPS_BASE = base
        break

if not INTERGPS_BASE:
    logger.error("Inter-GPS not found in any expected location!")
    logger.error(f"Tried: {possible_bases}")
else:
    logger.info(f"Using Inter-GPS at: {INTERGPS_BASE}")
    # Add Inter-GPS to Python path
    sys.path.insert(0, os.path.join(INTERGPS_BASE, "symbolic_solver"))
    sys.path.insert(0, INTERGPS_BASE)

# Import Inter-GPS modules
try:
    from extended_definition import ExtendedDefinition
    from logic_parser import LogicParser
    from logic_solver import LogicSolver
    logger.info("Successfully imported Inter-GPS modules")
except ImportError as e:
    logger.error(f"Failed to import Inter-GPS modules: {e}")


class InterGPSTool(ToolBase):
    """Inter-GPS工具 - 几何问题形式化求解器"""
    
    name = "inter_gps"
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(name=self.name)
        
        # 工具描述
        self.description = "Formal geometry problem solver using Inter-GPS"
        self.parameters = {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "enum": ["solve", "formalize", "verify"],
                    "description": "Task type"
                },
                "problem_id": {
                    "type": "string",
                    "description": "Problem ID"
                },
                "text_logic": {
                    "type": "array",
                    "description": "Text logic forms"
                },
                "diagram_logic": {
                    "type": "array", 
                    "description": "Diagram logic forms"
                },
                "strategy": {
                    "type": "string",
                    "enum": ["final", "predict", "random", "low-first"],
                    "default": "low-first",
                    "description": "Search strategy"
                }
            },
            "required": ["task"]
        }
        
        self.config = config or {}
        self.debug = self.config.get("debug", False)
        
        # Inter-GPS配置
        self.time_limit = self.config.get("time_limit", 100)
        self.step_limit = self.config.get("step_limit", 100)
        self.strategy = self.config.get("strategy", "low-first")
        
        # 当前问题数据
        self.current_problem = None
        
    def reset(self, image=None):
        """重置工具状态"""
        self.current_problem = None
        logger.debug("Inter-GPS tool reset")
        
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行几何问题求解"""
        try:
            # 解析参数
            if isinstance(params, str):
                params = json.loads(params)
            
            task = params.get("task", "solve")
            
            # 获取逻辑形式
            text_logic, diagram_logic = self._get_logic_forms(params)
            
            if not text_logic or not diagram_logic:
                return {
                    "error": "Missing logic forms",
                    "error_type": "InvalidInput",
                    "success": False
                }
            
            # 根据任务类型执行
            if task == "solve":
                strategy = params.get("strategy", self.strategy)
                return self._solve_problem(text_logic, diagram_logic, strategy)
            elif task == "formalize":
                return self._formalize_problem(text_logic, diagram_logic)
            elif task == "verify":
                answer = params.get("answer", "")
                return self._verify_answer(text_logic, diagram_logic, answer)
            else:
                return {
                    "error": f"Unknown task: {task}",
                    "error_type": "InvalidTask",
                    "success": False
                }
                
        except Exception as e:
            logger.error(f"Inter-GPS execution error: {str(e)}")
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "success": False
            }
    
    def _get_logic_forms(self, params: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """获取逻辑形式"""
        # 优先使用直接提供的逻辑形式
        if "text_logic" in params and "diagram_logic" in params:
            return params["text_logic"], params["diagram_logic"]
        
        # 从当前问题上下文获取
        if self.current_problem:
            metadata = self.current_problem.get("metadata", {})
            text_logic = metadata.get("text_logic_form", [])
            diagram_logic = metadata.get("diagram_logic_form", [])
            return text_logic, diagram_logic
        
        return [], []
    
    def _solve_problem(self, text_logic: List[str], diagram_logic: List[str], strategy: str) -> Dict[str, Any]:
        """使用Inter-GPS求解几何问题"""
        if not INTERGPS_BASE:
            return {
                "error": "Inter-GPS not found",
                "error_type": "ConfigError",
                "success": False
            }
            
        try:
            # 准备数据结构
            text_parser = {"text_logic_forms": text_logic}
            diagram_parser = self._prepare_diagram_parser(diagram_logic)
            
            # 确定搜索策略
            enable_low_first = strategy in ["low-first", "final"]
            order_lst = None  # 不使用预测的定理序列
            
            # 创建求解参数
            class Args:
                debug_mode = self.debug
                enable_round = False
                round_limit = 10
                step_limit = self.step_limit
                time_limit = self.time_limit
                low_first = enable_low_first
            
            args = Args()
            
            # 执行求解
            start_time = time.time()
            
            try:
                # 使用超时控制
                target, answer, steps, step_lst = func_timeout(
                    self.time_limit,
                    self._solve_one_problem,
                    kwargs=dict(
                        args=args,
                        text_parser=text_parser,
                        diagram_parser=diagram_parser,
                        order_lst=order_lst
                    )
                )
            except FunctionTimedOut:
                return {
                    "success": False,
                    "error": f"Solving timeout after {self.time_limit} seconds",
                    "error_type": "Timeout"
                }
            except Exception as e:
                logger.error(f"Solving error: {str(e)}")
                return {
                    "success": False,
                    "error": str(e),
                    "error_type": "SolveError"
                }
            
            solve_time = time.time() - start_time
            
            # 处理结果
            if answer is not None:
                # 格式化答案
                if isinstance(answer, (int, float)):
                    answer_str = self._format_answer(answer)
                else:
                    answer_str = str(answer)
                
                return {
                    "success": True,
                    "solution": answer_str,
                    "method": f"Inter-GPS ({strategy})",
                    "solve_time": solve_time,
                    "steps": steps,
                    "target": str(target) if target else None
                }
            else:
                return {
                    "success": False,
                    "error": "No solution found",
                    "solve_time": solve_time,
                    "strategy": strategy
                }
                
        except Exception as e:
            logger.error(f"GPS solve error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "SolveError"
            }
    
    def _prepare_diagram_parser(self, diagram_logic: List[str]) -> Dict[str, Any]:
        """准备图表解析器数据"""
        # 提取点、线、圆等信息
        points = set()
        lines = []
        circles = []
        
        for logic in diagram_logic:
            # 提取所有大写字母作为点
            points.update(re.findall(r'\b[A-Z]\b', logic))
            
            # 提取线段
            line_matches = re.findall(r'Line\(([A-Z]), ([A-Z])\)', logic)
            for p1, p2 in line_matches:
                line = f"{p1}{p2}"
                if line not in lines and f"{p2}{p1}" not in lines:
                    lines.append(line)
            
            # 提取圆
            circle_matches = re.findall(r'Circle\(([A-Z])\)', logic)
            circles.extend(circle_matches)
        
        # 创建点位置（简单的布局）
        point_positions = {}
        sorted_points = sorted(list(points))
        for i, point in enumerate(sorted_points):
            # 简单的圆形布局
            angle = 2 * 3.14159 * i / len(sorted_points)
            x = 50 + 40 * math.cos(angle)
            y = 50 + 40 * math.sin(angle)
            point_positions[point] = [x, y]
        
        return {
            "point_positions": point_positions,
            "line_instances": lines,
            "circle_instances": circles,
            "diagram_logic_forms": diagram_logic
        }
    
    def _solve_one_problem(self, args, text_parser, diagram_parser, order_lst):
        """核心求解函数（来自Inter-GPS的test.py）"""
        import math
        
        # Set up the logic parser
        parser = LogicParser(ExtendedDefinition(debug=args.debug_mode))
        
        if diagram_parser is not None:
            # Define diagram primitive elements
            parser.logic.point_positions = diagram_parser['point_positions']
            
            isLetter = lambda ch: ch.upper() and len(ch) == 1
            parser.logic.define_point([_ for _ in parser.logic.point_positions if isLetter(_)])
            
            if args.debug_mode:
                logger.debug(f"Points: {parser.logic.point_positions}")
            
            lines = diagram_parser['line_instances']
            for line in lines:
                line = line.strip()
                if len(line) == 2 and isLetter(line[0]) and isLetter(line[1]):
                    parser.logic.define_line(line[0], line[1])
            
            circles = diagram_parser['circle_instances']
            for point in circles:
                parser.logic.define_circle(point)
            
            # Parse diagram logic forms
            logic_forms = diagram_parser['diagram_logic_forms']
            logic_forms = sorted(logic_forms, key=lambda x: x.find("Perpendicular") != -1)
            
            for logic_form in logic_forms:
                if logic_form.strip() != "":
                    if args.debug_mode:
                        logger.debug(f"Diagram logic: {logic_form}")
                    try:
                        parse_tree = parser.parse(logic_form)
                        parser.dfsParseTree(parse_tree)
                    except Exception as e:
                        if args.debug_mode:
                            logger.error(f"Parse error: {repr(e)}")
        
        # Parse text logic forms
        target = None
        text_logic_forms = text_parser["text_logic_forms"]
        for text in text_logic_forms:
            if args.debug_mode:
                logger.debug(f"Text logic: {text}")
            if text.find('Find') != -1:
                target = parser.findTarget(parser.parse(text))
            else:
                res = parser.parse(text)
                parser.dfsParseTree(res)
        
        if args.debug_mode:
            logger.debug(f"Target: {target}")
        
        # Set up, initialize and run the logic solver
        solver = LogicSolver(parser.logic)
        solver.initSearch()
        
        answer, steps, step_lst = solver.Search(
            target=target,
            order_list=order_lst,
            round_or_step=args.enable_round,
            upper_bound=args.round_limit if args.enable_round else args.step_limit,
            enable_low_first=args.low_first
        )
        
        return target, answer, steps, step_lst
    
    def _format_answer(self, answer, ndigits: int = 2) -> str:
        """格式化答案"""
        try:
            num = float(answer)
            if num == int(num):
                return str(int(num))
            return f"{num:.{ndigits}f}".rstrip('0').rstrip('.')
        except:
            return str(answer)
    
    def _formalize_problem(self, text_logic: List[str], diagram_logic: List[str]) -> Dict[str, Any]:
        """形式化分析问题结构"""
        # 提取问题元素
        points = set()
        lines = []
        angles = []
        triangles = []
        constraints = []
        relationships = []
        variables = set()
        
        all_logic = text_logic + diagram_logic
        
        for logic in all_logic:
            # 提取点
            points.update(re.findall(r'\b[A-Z]\b', logic))
            
            # 提取线段
            line_matches = re.findall(r'Line\(([A-Z]), ([A-Z])\)', logic)
            lines.extend(line_matches)
            
            # 提取角度
            angle_matches = re.findall(r'Angle\(([A-Z]), ([A-Z]), ([A-Z])\)', logic)
            angles.extend(angle_matches)
            
            # 提取三角形
            triangle_matches = re.findall(r'Triangle\(([A-Z]),\s*([A-Z]),\s*([A-Z])\)', logic)
            triangles.extend(triangle_matches)
            
            # 提取变量
            variables.update(re.findall(r'\b([a-z])\b', logic))
            
            # 分类逻辑形式
            if 'Equals' in logic:
                constraints.append(logic)
            elif any(rel in logic for rel in ['Congruent', 'Parallel', 'Perpendicular', 'Similar']):
                relationships.append(logic)
        
        return {
            "success": True,
            "elements": {
                "points": sorted(list(points)),
                "lines": lines,
                "angles": angles,
                "triangles": triangles,
                "variables": sorted(list(variables))
            },
            "logic_forms": {
                "text": text_logic,
                "diagram": diagram_logic,
                "constraints": constraints,
                "relationships": relationships
            },
            "statistics": {
                "num_points": len(points),
                "num_lines": len(lines),
                "num_angles": len(angles),
                "num_triangles": len(triangles),
                "num_variables": len(variables),
                "num_constraints": len(constraints),
                "num_relationships": len(relationships)
            }
        }
    
    def _verify_answer(self, text_logic: List[str], diagram_logic: List[str], answer: str) -> Dict[str, Any]:
        """验证答案是否正确"""
        # 求解问题
        result = self._solve_problem(text_logic, diagram_logic, self.strategy)
        
        if result.get("success"):
            computed = result.get("solution", "")
            is_correct = self._answers_match(computed, answer)
            
            return {
                "success": True,
                "is_correct": is_correct,
                "computed_answer": computed,
                "provided_answer": answer,
                "match_details": {
                    "numeric_match": self._numeric_match(computed, answer),
                    "string_match": str(computed).strip() == str(answer).strip()
                }
            }
        else:
            return {
                "success": False,
                "error": "Could not verify - solving failed",
                "provided_answer": answer,
                "solve_attempt": result
            }
    
    def _answers_match(self, pred: str, gold: str, rel_tol: float = 1e-9, abs_tol: float = 0.01) -> bool:
        """比较两个答案是否相等"""
        # 先尝试数值比较
        if self._numeric_match(pred, gold, rel_tol, abs_tol):
            return True
        
        # 字符串比较
        return str(pred).strip().lower() == str(gold).strip().lower()
    
    def _numeric_match(self, pred: str, gold: str, rel_tol: float = 1e-9, abs_tol: float = 0.01) -> bool:
        """数值比较"""
        try:
            p = float(str(pred).strip())
            g = float(str(gold).strip())
            
            # 绝对误差或相对误差
            return (abs(p - g) < abs_tol or 
                    abs(p - g) / max(abs(p), abs(g), 1) < rel_tol)
        except:
            return False
    
    def set_current_problem(self, problem_data: Dict[str, Any]):
        """设置当前问题（由环境调用）"""
        self.current_problem = problem_data