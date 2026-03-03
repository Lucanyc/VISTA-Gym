# /vlm_gym/environments/tools/geometry_tools/sympy_geometry.py
"""
SymPy几何计算工具
用于处理各种几何计算任务
"""
import json
import math
from typing import Dict, Any, List, Tuple, Optional
import logging

from sympy import Point, pi, sqrt, sin, cos, tan, asin, acos, atan2
from sympy.geometry import (
    Triangle, Line, Polygon, Ray, Segment, Circle
)
from sympy.geometry.util import intersection  # 正确的导入路径
from sympy import symbols, solve, Eq, simplify

from vlm_gym.environments.tools.base import ToolBase


class SympyGeometryTool(ToolBase):
    """SymPy几何计算工具 - 提供精确的几何计算功能"""
    
    # 必须定义name属性以触发自动注册
    name = "sympy_geometry"
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化SymPy几何工具
        
        Args:
            config: 配置字典
        """
        # 设置工具属性
        self.name = "sympy_geometry"
        self.description = "精确几何计算工具，支持三角形、圆、多边形等计算"
        self.capabilities = [
            "三角形角度计算",
            "圆的相关计算",
            "多边形面积和周长",
            "几何关系判断",
            "坐标计算"
        ]
        
        # 传递name参数给父类
        super().__init__(name=self.name)
        
        # 配置
        self.config = config or {}
        self.precision = self.config.get('precision', 4)  # 小数位数
        
        # 设置日志
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # 当前上下文（可选：存储图像相关的几何信息）
        self.current_context = None
        
    def reset(self, context: Any = None):
        """重置工具状态
        
        Args:
            context: 可选的上下文信息（如图像中检测到的几何元素）
        """
        self.current_context = context
        self.logger.info("SymPy Geometry tool reset")
        
    def execute(self, action_string: str) -> Dict[str, Any]:
        """执行几何计算
        
        Args:
            action_string: JSON格式的动作字符串，包含：
                - function: 要调用的函数名
                - args: 函数参数
                
        Returns:
            计算结果字典
        """
        try:
            # 解析参数
            if isinstance(action_string, str):
                params = json.loads(action_string)
            else:
                params = action_string
                
            function = params.get('function', '')
            args = params.get('args', {})
            
            # 根据函数名调用相应方法
            if hasattr(self, f'_calc_{function}'):
                method = getattr(self, f'_calc_{function}')
                result = method(**args)
                return {
                    "success": True,
                    "function": function,
                    "result": result,
                    "formatted": self._format_result(result)
                }
            else:
                available_functions = [
                    f.replace('_calc_', '') for f in dir(self) 
                    if f.startswith('_calc_')
                ]
                return {
                    "error": f"Unknown function: {function}",
                    "available_functions": available_functions
                }
                
        except json.JSONDecodeError as e:
            return {
                "error": f"Invalid JSON: {str(e)}",
                "error_type": "JSONDecodeError"
            }
        except Exception as e:
            import traceback
            return {
                "error": f"Calculation failed: {str(e)}",
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
    
    # ===== 三角形相关计算 =====
    
    def _calc_triangle_angle(self, A: List[float], B: List[float], C: List[float], 
                             vertex: str = 'A') -> Dict[str, Any]:
        """计算三角形的内角
        
        Args:
            A, B, C: 三角形顶点坐标 [x, y]
            vertex: 要计算的角的顶点 ('A', 'B', 或 'C')
            
        Returns:
            角度信息（度和弧度）
        """
        points = {
            'A': Point(*A),
            'B': Point(*B),
            'C': Point(*C)
        }
        
        tri = Triangle(points['A'], points['B'], points['C'])
        
        # 获取指定顶点的角度
        angle_rad = tri.angles[points[vertex]]
        angle_deg = float(angle_rad * 180 / pi)
        
        return {
            "angle_degrees": round(angle_deg, self.precision),
            "angle_radians": round(float(angle_rad), self.precision),
            "vertex": vertex
        }
    
    def _calc_triangle_all_angles(self, A: List[float], B: List[float], 
                                  C: List[float]) -> Dict[str, Any]:
        """计算三角形的所有内角"""
        result = {}
        for vertex in ['A', 'B', 'C']:
            angle_info = self._calc_triangle_angle(A, B, C, vertex)
            result[f"angle_{vertex}"] = angle_info["angle_degrees"]
            
        # 验证和为180度
        angle_sum = sum(result.values())
        result["angle_sum"] = round(angle_sum, self.precision)
        result["is_valid"] = abs(angle_sum - 180) < 0.01
        
        return result
    
    def _calc_triangle_area(self, A: List[float], B: List[float], 
                           C: List[float]) -> Dict[str, Any]:
        """计算三角形面积"""
        tri = Triangle(Point(*A), Point(*B), Point(*C))
        area = float(tri.area)
        perimeter = float(tri.perimeter)
        
        return {
            "area": round(area, self.precision),
            "perimeter": round(perimeter, self.precision)
        }
    
    def _calc_triangle_type(self, A: List[float], B: List[float], 
                           C: List[float]) -> Dict[str, Any]:
        """判断三角形类型"""
        tri = Triangle(Point(*A), Point(*B), Point(*C))
        
        # 计算边长
        sides = [
            float(tri.sides[0].length),
            float(tri.sides[1].length),
            float(tri.sides[2].length)
        ]
        sides.sort()
        
        # 判断类型
        is_equilateral = tri.is_equilateral()
        is_isosceles = tri.is_isosceles()
        is_scalene = tri.is_scalene()
        is_right = tri.is_right()
        
        # 通过边长关系判断
        a, b, c = sides
        is_acute = a**2 + b**2 > c**2
        is_obtuse = a**2 + b**2 < c**2
        
        return {
            "is_equilateral": is_equilateral,
            "is_isosceles": is_isosceles,
            "is_scalene": is_scalene,
            "is_right": is_right,
            "is_acute": is_acute and not is_right,
            "is_obtuse": is_obtuse,
            "sides": [round(s, self.precision) for s in sides]
        }
    
    # ===== 圆相关计算 =====
    
    def _calc_circle_from_points(self, p1: List[float], p2: List[float], 
                                 p3: List[float]) -> Dict[str, Any]:
        """通过三点确定圆"""
        try:
            # 创建通过三点的圆
            circle = Circle(Point(*p1), Point(*p2), Point(*p3))
            center = circle.center
            radius = float(circle.radius)
            
            return {
                "center": [float(center.x), float(center.y)],
                "radius": round(radius, self.precision),
                "area": round(float(pi * radius**2), self.precision),
                "circumference": round(float(2 * pi * radius), self.precision)
            }
        except Exception as e:
            return {
                "error": "Points are collinear or invalid",
                "message": str(e)
            }
    
    def _calc_inscribed_angle(self, center: List[float], radius: float,
                             angle_degrees: float) -> Dict[str, Any]:
        """计算圆周角和圆心角的关系
        
        圆周角定理：圆周角是圆心角的一半
        """
        inscribed = angle_degrees
        central = inscribed * 2
        
        # 计算弧长
        angle_rad = math.radians(central)
        arc_length = radius * angle_rad
        
        # 计算扇形面积
        sector_area = 0.5 * radius**2 * angle_rad
        
        return {
            "inscribed_angle": inscribed,
            "central_angle": central,
            "arc_length": round(arc_length, self.precision),
            "sector_area": round(sector_area, self.precision)
        }
    
    # ===== 多边形计算 =====
    
    def _calc_polygon_area(self, vertices: List[List[float]]) -> Dict[str, Any]:
        """计算多边形面积和周长"""
        points = [Point(*v) for v in vertices]
        poly = Polygon(*points)
        
        area = float(poly.area)
        perimeter = float(poly.perimeter)
        
        # 判断是否为凸多边形
        is_convex = poly.is_convex()
        
        return {
            "area": round(abs(area), self.precision),
            "perimeter": round(perimeter, self.precision),
            "num_vertices": len(vertices),
            "is_convex": is_convex
        }
    
    # ===== 线段和角度计算 =====
    
    def _calc_angle_between_lines(self, p1: List[float], p2: List[float],
                                  p3: List[float], p4: List[float]) -> Dict[str, Any]:
        """计算两条线段之间的夹角
        
        线段1: p1 -> p2
        线段2: p3 -> p4
        """
        line1 = Line(Point(*p1), Point(*p2))
        line2 = Line(Point(*p3), Point(*p4))
        
        # 计算夹角（弧度）- 使用 angle_between 方法
        angle_rad = line1.angle_between(line2)
        angle_deg = float(angle_rad * 180 / pi)
        
        # 判断是否平行或垂直
        is_parallel = line1.is_parallel(line2)
        is_perpendicular = line1.is_perpendicular(line2)
        
        return {
            "angle_degrees": round(angle_deg, self.precision),
            "angle_radians": round(float(angle_rad), self.precision),
            "is_parallel": is_parallel,
            "is_perpendicular": is_perpendicular
        }
    
    def _calc_distance_point_to_line(self, point: List[float], 
                                    line_p1: List[float], 
                                    line_p2: List[float]) -> Dict[str, Any]:
        """计算点到直线的距离"""
        p = Point(*point)
        line = Line(Point(*line_p1), Point(*line_p2))
        
        distance = float(line.distance(p))
        
        # 找到垂足
        perpendicular_line = line.perpendicular_line(p)
        foot = intersection(line, perpendicular_line)[0]
        
        return {
            "distance": round(distance, self.precision),
            "foot_of_perpendicular": [float(foot.x), float(foot.y)]
        }
    
    # ===== 实用几何计算 =====
    
    def _calc_triangle_angle_from_two(self, angle1: float, angle2: float) -> Dict[str, Any]:
        """已知三角形两个角，求第三个角"""
        angle3 = 180 - angle1 - angle2
        
        if angle3 <= 0 or angle3 >= 180:
            return {
                "error": "Invalid angles - sum must be less than 180",
                "given_angles": [angle1, angle2],
                "sum": angle1 + angle2
            }
            
        return {
            "angle1": angle1,
            "angle2": angle2,
            "angle3": round(angle3, self.precision),
            "sum": 180.0
        }
    
    def _calc_pythagorean(self, a: float = None, b: float = None, 
                         c: float = None) -> Dict[str, Any]:
        """勾股定理计算
        
        a² + b² = c²
        给定任意两边，求第三边
        """
        if sum([a is not None, b is not None, c is not None]) != 2:
            return {
                "error": "Exactly two sides must be provided"
            }
            
        try:
            if c is None:
                # 求斜边
                c = math.sqrt(a**2 + b**2)
                result_type = "hypotenuse"
            elif a is None:
                # 求直角边a
                a = math.sqrt(c**2 - b**2)
                result_type = "leg_a"
            else:
                # 求直角边b
                b = math.sqrt(c**2 - a**2)
                result_type = "leg_b"
                
            return {
                "a": round(a, self.precision),
                "b": round(b, self.precision),
                "c": round(c, self.precision),
                "calculated": result_type
            }
        except ValueError:
            return {
                "error": "Invalid values - cannot form a right triangle"
            }
    
    # ===== 辅助方法 =====
    
    def _format_result(self, result: Dict[str, Any]) -> str:
        """格式化结果为易读的字符串"""
        if "error" in result:
            return f"Error: {result['error']}"
            
        # 根据不同的结果类型格式化
        formatted_parts = []
        
        # 角度相关
        if "angle_degrees" in result:
            formatted_parts.append(f"Angle: {result['angle_degrees']}°")
        if "angle_radians" in result:
            formatted_parts.append(f"({result['angle_radians']} rad)")
            
        # 面积和周长
        if "area" in result:
            formatted_parts.append(f"Area: {result['area']}")
        if "perimeter" in result:
            formatted_parts.append(f"Perimeter: {result['perimeter']}")
            
        # 距离
        if "distance" in result:
            formatted_parts.append(f"Distance: {result['distance']}")
            
        # 如果没有特定格式，返回JSON
        if not formatted_parts:
            return json.dumps(result, indent=2)
            
        return " | ".join(formatted_parts)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """返回工具能力描述"""
        functions = [
            {
                "name": "triangle_angle",
                "description": "计算三角形的内角",
                "parameters": {
                    "A": "顶点A坐标 [x, y]",
                    "B": "顶点B坐标 [x, y]",
                    "C": "顶点C坐标 [x, y]",
                    "vertex": "要计算的角的顶点 (A/B/C)"
                }
            },
            {
                "name": "triangle_area",
                "description": "计算三角形面积和周长",
                "parameters": {
                    "A": "顶点A坐标 [x, y]",
                    "B": "顶点B坐标 [x, y]",
                    "C": "顶点C坐标 [x, y]"
                }
            },
            {
                "name": "circle_from_points",
                "description": "通过三点确定圆",
                "parameters": {
                    "p1": "第一个点 [x, y]",
                    "p2": "第二个点 [x, y]",
                    "p3": "第三个点 [x, y]"
                }
            },
            {
                "name": "inscribed_angle",
                "description": "计算圆周角和圆心角关系",
                "parameters": {
                    "center": "圆心坐标 [x, y]",
                    "radius": "半径",
                    "angle_degrees": "圆周角（度）"
                }
            },
            {
                "name": "pythagorean",
                "description": "勾股定理计算",
                "parameters": {
                    "a": "直角边a（可选）",
                    "b": "直角边b（可选）",
                    "c": "斜边c（可选）"
                }
            }
        ]
        
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "functions": functions,
            "precision": self.precision
        }


# 测试代码
if __name__ == "__main__":
    # 创建工具实例
    tool = SympyGeometryTool()
    
    # 测试1: 计算三角形角度
    print("=== 测试1: 三角形角度计算 ===")
    tool.reset()
    result = tool.execute(json.dumps({
        "function": "triangle_angle",
        "args": {
            "A": [0, 0],
            "B": [3, 0],
            "C": [0, 4],
            "vertex": "A"
        }
    }))
    print(result)
    
    # 测试2: 勾股定理
    print("\n=== 测试2: 勾股定理 ===")
    result = tool.execute(json.dumps({
        "function": "pythagorean",
        "args": {
            "a": 3,
            "b": 4
        }
    }))
    print(result)
    
    # 测试3: 圆周角
    print("\n=== 测试3: 圆周角计算 ===")
    result = tool.execute(json.dumps({
        "function": "inscribed_angle",
        "args": {
            "center": [0, 0],
            "radius": 5,
            "angle_degrees": 30
        }
    }))
    print(result)