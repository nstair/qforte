#!/usr/bin/env python3
"""
Script to automatically update qforte.pyi type stub file from bindings.cc
This script parses the pybind11 bindings and updates the existing type stubs.
"""

import re
import os
import sys
from pathlib import Path


def extract_bindings_info(bindings_path):
    """Extract class and method information from bindings.cc"""
    
    with open(bindings_path, 'r') as f:
        content = f.read()
    
    classes_info = {}
    
    # Find all py::class_ declarations
    class_pattern = r'py::class_<([^>]+)>\(m,\s*"([^"]+)"\)(.*?)(?=(?:py::class_|py::enum_|m\.def)|(?:\n\s*m\.def))'
    
    for match in re.finditer(class_pattern, content, re.DOTALL):
        cpp_types = match.group(1).strip()
        python_name = match.group(2).strip()
        class_body = match.group(3)
        
        # Extract the primary C++ class name (first type in template)
        cpp_class = cpp_types.split(',')[0].strip()
        
        # Extract constructor signatures
        constructors = []
        init_pattern = r'\.def\(py::init<([^>]*)>\(\)[^,]*(?:,\s*"([^"]*)"[^,]*)*(?:,\s*"([^"]*)")*[^)]*\)'
        for init_match in re.finditer(init_pattern, class_body):
            params = init_match.group(1).strip()
            constructors.append(params)
        
        # Extract method definitions
        methods = {}
        def_pattern = r'\.def\(\s*"([^"]+)",\s*&[^:]+::([^,)]+)(?:,\s*py::arg\([^)]+\))*(?:,\s*"([^"]*)")*[^)]*\)'
        
        for def_match in re.finditer(def_pattern, class_body):
            method_name = def_match.group(1)
            cpp_method = def_match.group(2)
            
            if method_name not in methods:
                methods[method_name] = []
            methods[method_name].append(cpp_method)
        
        # Extract static methods
        static_methods = {}
        static_pattern = r'\.def_static\(\s*"([^"]+)",\s*&[^:]+::([^,)]+)'
        for static_match in re.finditer(static_pattern, class_body):
            method_name = static_match.group(1)
            cpp_method = static_match.group(2)
            static_methods[method_name] = cpp_method
        
        classes_info[python_name] = {
            'cpp_class': cpp_class,
            'constructors': constructors,
            'methods': methods,
            'static_methods': static_methods
        }
    
    # Extract standalone functions
    functions = []
    func_pattern = r'm\.def\(\s*"([^"]+)"'
    for func_match in re.finditer(func_pattern, content):
        functions.append(func_match.group(1))
    
    return classes_info, functions


def update_pyi_file(pyi_path, classes_info, functions):
    """Update the existing .pyi file with information from bindings"""
    
    # Read the current .pyi file
    with open(pyi_path, 'r') as f:
        current_content = f.read()
    
    print("Classes found in bindings:")
    for class_name, info in classes_info.items():
        print(f"  {class_name} ({info['cpp_class']})")
        print(f"    Constructors: {len(info['constructors'])}")
        print(f"    Methods: {len(info['methods'])}")
        print(f"    Static methods: {len(info['static_methods'])}")
    
    print(f"\nFunctions found in bindings: {functions}")
    
    return current_content  # For now, just return the current content


def verify_pyi_completeness(pyi_path, classes_info):
    """Verify that the .pyi file contains all classes and methods from bindings"""
    
    with open(pyi_path, 'r') as f:
        pyi_content = f.read()
    
    missing_classes = []
    incomplete_classes = {}
    
    for class_name, info in classes_info.items():
        # Check if class is defined in .pyi
        class_pattern = rf'class {re.escape(class_name)}\s*:'
        if not re.search(class_pattern, pyi_content):
            missing_classes.append(class_name)
            continue
        
        # Check methods for this class
        missing_methods = []
        
        # Extract the class section from .pyi
        class_start = re.search(rf'class {re.escape(class_name)}\s*:', pyi_content)
        if class_start:
            remaining_content = pyi_content[class_start.end():]
            next_class = re.search(r'\nclass \w+\s*:', remaining_content)
            if next_class:
                class_content = remaining_content[:next_class.start()]
            else:
                class_content = remaining_content
            
            for method_name in info['methods']:
                method_pattern = rf'def {re.escape(method_name)}\s*\('
                if not re.search(method_pattern, class_content):
                    missing_methods.append(method_name)
        
        if missing_methods:
            incomplete_classes[class_name] = missing_methods
    
    # Report findings
    if missing_classes:
        print(f"\nMissing classes in .pyi file:")
        for cls in missing_classes:
            print(f"  - {cls}")
    
    if incomplete_classes:
        print(f"\nClasses with missing methods:")
        for cls, methods in incomplete_classes.items():
            print(f"  {cls}:")
            for method in methods:
                print(f"    - {method}")
    
    if not missing_classes and not incomplete_classes:
        print("\n All classes and methods from bindings are present in .pyi file")
    
    return missing_classes, incomplete_classes


def main():
    """Main function"""
    
    # Get paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    bindings_path = project_root / "src" / "qforte" / "bindings.cc"
    pyi_path = project_root / "src" / "qforte" / "qforte.pyi"
    
    if not bindings_path.exists():
        print(f"Error: bindings.cc not found at {bindings_path}")
        sys.exit(1)
    
    if not pyi_path.exists():
        print(f"Error: qforte.pyi not found at {pyi_path}")
        sys.exit(1)
    
    print(f"Analyzing bindings: {bindings_path}")
    classes_info, functions = extract_bindings_info(bindings_path)
    
    print(f"Verifying completeness of: {pyi_path}")
    missing_classes, incomplete_classes = verify_pyi_completeness(pyi_path, classes_info)
    
    # Suggest improvements
    if missing_classes or incomplete_classes:
        print(f"\nTo fix these issues, you can:")
        print(f"1. Manually add missing classes/methods to {pyi_path}")
        print(f"2. Use this script's output to guide your updates")
        print(f"3. Integrate this script into your CMake build process")


if __name__ == "__main__":
    main()
