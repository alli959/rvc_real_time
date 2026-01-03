#!/usr/bin/env python3
"""
Simple validation script to test the application structure
without requiring all dependencies.
"""

import sys
import os
from pathlib import Path

def test_structure():
    """Test that all required files and directories exist"""
    print("Testing project structure...")
    
    required_files = [
        'main.py',
        'requirements.txt',
        'Dockerfile',
        'docker-compose.yml',
        'README.md',
        '.gitignore',
        'setup.sh',
        'app/__init__.py',
        'app/audio_stream.py',
        'app/feature_extraction.py',
        'app/model_manager.py',
        'app/chunk_processor.py',
        'app/streaming_api.py',
        'app/config.py',
        'examples/websocket_client.py',
        'examples/socket_client.py',
    ]
    
    required_dirs = [
        'app',
        'assets',
        'assets/models',
        'examples',
    ]
    
    all_ok = True
    
    # Check directories
    for dir_path in required_dirs:
        if not Path(dir_path).is_dir():
            print(f"❌ Missing directory: {dir_path}")
            all_ok = False
        else:
            print(f"✓ Directory exists: {dir_path}")
    
    # Check files
    for file_path in required_files:
        if not Path(file_path).is_file():
            print(f"❌ Missing file: {file_path}")
            all_ok = False
        else:
            print(f"✓ File exists: {file_path}")
    
    return all_ok

def test_imports():
    """Test that modules can be imported (without dependencies)"""
    print("\nTesting Python module structure...")
    
    # Just check syntax, don't actually import (to avoid dependency issues)
    import py_compile
    
    python_files = [
        'main.py',
        'app/__init__.py',
        'app/audio_stream.py',
        'app/feature_extraction.py',
        'app/model_manager.py',
        'app/chunk_processor.py',
        'app/streaming_api.py',
        'app/config.py',
    ]
    
    all_ok = True
    for file_path in python_files:
        try:
            py_compile.compile(file_path, doraise=True)
            print(f"✓ Syntax valid: {file_path}")
        except py_compile.PyCompileError as e:
            print(f"❌ Syntax error in {file_path}: {e}")
            all_ok = False
    
    return all_ok

def test_documentation():
    """Test that documentation exists and has minimum content"""
    print("\nTesting documentation...")
    
    all_ok = True
    
    # Check main README
    readme_path = Path('README.md')
    if readme_path.exists():
        content = readme_path.read_text()
        required_sections = [
            'Features',
            'Installation',
            'Usage',
            'API',
            'Docker',
        ]
        
        for section in required_sections:
            if section.lower() in content.lower():
                print(f"✓ README contains '{section}' section")
            else:
                print(f"❌ README missing '{section}' section")
                all_ok = False
    else:
        print("❌ README.md not found")
        all_ok = False
    
    return all_ok

def main():
    print("=" * 60)
    print("RVC Real-time Voice Conversion - Validation Test")
    print("=" * 60)
    print()
    
    results = []
    
    results.append(("Structure", test_structure()))
    results.append(("Python Syntax", test_imports()))
    results.append(("Documentation", test_documentation()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All validation tests passed!")
        return 0
    else:
        print("\n❌ Some validation tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
