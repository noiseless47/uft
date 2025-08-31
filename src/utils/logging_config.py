"""
Logging configuration for the football squad selection pipeline.
"""

import logging
import os
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: str = "logs/pipeline.log"):
    """Setup logging configuration."""
    
    # Create logs directory
    log_dir = Path(log_file).parent
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
