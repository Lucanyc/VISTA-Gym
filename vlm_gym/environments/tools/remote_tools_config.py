# 远程工具 API 配置
REMOTE_TOOL_APIS = {
    "multimath": {
        "url": "http://localhost:8001",
        "timeout": 60,
        "enabled": True
    },
    "intergps": {
        "url": "http://localhost:8002", 
        "timeout": 120,
        "enabled": False  # 暂未部署
    },
    "chartmoe": {
        "url": "http://localhost:8003",
        "timeout": 60,
        "enabled": False  # 暂未部署
    }
}