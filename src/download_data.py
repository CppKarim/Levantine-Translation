#!/usr/bin/env python3
"""
Simple file downloader - lightweight alternative to curl
"""

import os
import sys
import argparse
import urllib.request
from urllib.parse import urlparse

def download_file(url, output_path):
    """Download a file from a URL to the specified path."""
    try:
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Download the file
        print(f"Downloading {url} to {output_path}...")
        urllib.request.urlretrieve(url, output_path)
        print("Download complete!")
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def get_filename_from_url(url):
    """Extract the filename from a URL."""
    path = urlparse(url).path
    return os.path.basename(path) or "download"

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download a file from the internet")
    parser.add_argument("url", help="URL of the file to download")
    parser.add_argument("-o", "--output", help="Output filename (default: filename from URL)")
    parser.add_argument("-d", "--directory", default=".", help="Directory to save the file (default: current directory)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Use filename from URL
        filename = get_filename_from_url(args.url)
        output_path = os.path.join(args.directory, filename)
    
    # Download the file
    download_file(args.url, output_path)