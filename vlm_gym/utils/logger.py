import logging
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any
from datetime import datetime
import json
from logging.handlers import RotatingFileHandler

class ColoredFormatter(logging.Formatter):
    """Colored log formatter"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields if any
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
            
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)

def setup_logger(
    name: str = "vlm_gym",
    level: Union[str, int] = "INFO",
    log_dir: Optional[Union[str, Path]] = None,
    console: bool = True,
    file: bool = True,
    json_format: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """Setup logger with console and file handlers"""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if json_format:
            console_formatter = JSONFormatter()
        else:
            console_formatter = ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        
        if json_format:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_experiment_start(
    logger: logging.Logger,
    config: Dict[str, Any],
    experiment_id: str
):
    """Log experiment start with configuration"""
    logger.info("=" * 50)
    logger.info(f"Starting experiment: {experiment_id}")
    logger.info("=" * 50)
    logger.info("Configuration:")
    
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")
            
    logger.info("=" * 50)

def log_episode_summary(
    logger: logging.Logger,
    episode_num: int,
    summary: Dict[str, Any]
):
    """Log episode summary"""
    logger.info(f"Episode {episode_num} Summary:")
    logger.info(f"  Total Steps: {summary.get('total_steps', 0)}")
    logger.info(f"  Total Reward: {summary.get('total_reward', 0):.4f}")
    logger.info(f"  Success: {summary.get('success', False)}")
    
    if 'action_distribution' in summary:
        logger.info("  Action Distribution:")
        for action, count in summary['action_distribution'].items():
            logger.info(f"    {action}: {count}")

class LoggerContext:
    """Context manager for adding extra fields to logs"""
    
    def __init__(self, logger: logging.Logger, **kwargs):
        self.logger = logger
        self.extra_fields = kwargs
        self.old_factory = None
        
    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            record.extra_fields = self.extra_fields
            return record
            
        logging.setLogRecordFactory(record_factory)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)

# Create default logger
default_logger = setup_logger()
