"""
Utility Functions
=================

Logging, file handling, and helper functions.
"""

import os
import sys
import logging
from datetime import datetime
from typing import List, Optional


def setup_logging(log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger('predictive_maintenance')
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with simple format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler with detailed format
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def ensure_directories(*dirs: str) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        *dirs: Variable number of directory paths
    """
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def print_banner():
    """Print the project banner."""
    banner = """
================================================================================
                                                                                
              NASA C-MAPSS PREDICTIVE MAINTENANCE SYSTEM                    
                                                                                
              Remaining Useful Life (RUL) Prediction Pipeline               
                                                                                
================================================================================
    """
    print(banner)


def print_section(title: str, width: int = 80):
    """
    Print a formatted section header.
    
    Args:
        title: Section title
        width: Width of the separator line
    """
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")


def print_summary(start_time: datetime, metrics: dict):
    """
    Print pipeline execution summary.
    
    Args:
        start_time: Pipeline start timestamp
        metrics: Model performance metrics
    """
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print_section("PIPELINE EXECUTION SUMMARY")
    
    print(f"⏱  Execution Time: {duration:.2f} seconds")
    print(f" Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if metrics:
        print("\n Model Performance:")
        print(f"   • MAE:  {metrics.get('mae', 'N/A'):.4f}")
        print(f"   • RMSE: {metrics.get('rmse', 'N/A'):.4f}")
        print(f"   • R²:   {metrics.get('r2', 'N/A'):.4f}")
    
    print("\n Pipeline completed successfully!")


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def get_file_info(file_path: str) -> dict:
    """
    Get file information.
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file info
    """
    if not os.path.exists(file_path):
        return {}
    
    stat = os.stat(file_path)
    return {
        'path': file_path,
        'size': stat.st_size,
        'size_formatted': format_file_size(stat.st_size),
        'modified': datetime.fromtimestamp(stat.st_mtime)
    }


def list_output_files(output_dir: str) -> List[str]:
    """
    List all files in output directory.
    
    Args:
        output_dir: Output directory path
        
    Returns:
        List of file paths
    """
    if not os.path.exists(output_dir):
        return []
    
    files = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isfile(item_path):
            files.append(item_path)
    
    return sorted(files)
