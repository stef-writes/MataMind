"""
utils.py

This script provides helper functions for common tasks, such as path handling, data loading, plotting utilities, and other reusable code.
"""
import os

def create_directory_if_not_exists(directory):
    """
    Create a directory if it does not exist.
    Args:
        directory (str): Path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
