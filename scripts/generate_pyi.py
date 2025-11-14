#!/usr/bin/env python3
"""
Script to automatically generate qforte.pyi type stub file from bindings.cc
This script parses the pybind11 bindings and generates appropriate type stubs.
"""

import re
import os
import sys
from pathlib import Path


def parse_bindings_file(bindings_path):
    """Parse the bindings.cc file and extract class and method definitions."""
    
    with open(bindings_path, 'r') as f:
        content = f.read()
    
    # Extract class definitions
    classes = {}
    
    # Pattern to match py::class_ definitions
    class_pattern = r'py::class_<(\w+)>\(m,\s*"(\w+)"\)(.*?)(?=py::class_|py::enum_|m\.def|$)'
    
    for match in re.finditer(class_pattern, content, re.DOTALL):
        cpp_class = match.group(1)
        python_class = match.group(2)
        class_body = match.group(3)
        
        # Extract methods for this class
        methods = []
        
        # Pattern to match .def() calls
        def_pattern = r'\.def\(\s*"([^"]+)"[^)]*\)'
        for def_match in re.finditer(def_pattern, class_body):
            method_name = def_match.group(1)
            methods.append(method_name)
        
        # Pattern to match .def_static() calls
        static_pattern = r'\.def_static\(\s*"([^"]+)"[^)]*\)'
        static_methods = []
        for static_match in re.finditer(static_pattern, class_body):
            method_name = static_match.group(1)
            static_methods.append(method_name)
        
        classes[python_class] = {
            'cpp_class': cpp_class,
            'methods': methods,
            'static_methods': static_methods
        }
    
    # Extract standalone functions (m.def)
    functions = []
    function_pattern = r'm\.def\(\s*"([^"]+)"[^)]*\)'
    for func_match in re.finditer(function_pattern, content):
        function_name = func_match.group(1)
        functions.append(function_name)
    
    return classes, functions


def generate_pyi_content(classes, functions):
    """Generate the content of the .pyi file."""
    
    pyi_content = '''# qforte.pyi - Type stubs for the qforte quantum chemistry library
# Auto-generated from bindings.cc - DO NOT EDIT MANUALLY
from typing import List, Dict, Tuple, Union, Optional, Iterator, Any, overload
import numpy as np
from numpy.typing import NDArray

# Complex number type alias
Complex = Union[complex, float]

'''
    
    # Class definitions with basic type hints
    class_signatures = {
        'Circuit': {
            '__init__': 'def __init__(self) -> None: ...',
            'add': ['def add(self, gate: Gate) -> None: ...', 'def add(self, circuit: Circuit) -> None: ...'],
            'add_gate': 'def add_gate(self, gate: Gate) -> None: ...',
            'add_circuit': 'def add_circuit(self, circuit: Circuit) -> None: ...',
            'gates': 'def gates(self) -> List[Gate]: ...',
            'sparse_matrix': 'def sparse_matrix(self) -> SparseMatrix: ...',
            'size': 'def size(self) -> int: ...',
            'adjoint': 'def adjoint(self) -> Circuit: ...',
            'canonicalize_pauli_circuit': 'def canonicalize_pauli_circuit(self) -> None: ...',
            'set_parameters': 'def set_parameters(self, parameters: List[Complex]) -> None: ...',
            'get_num_cnots': 'def get_num_cnots(self) -> int: ...',
            'str': 'def str(self) -> str: ...',
            '__str__': 'def __str__(self) -> str: ...',
            '__repr__': 'def __repr__(self) -> str: ...'
        },
        'SQOperator': {
            '__init__': 'def __init__(self) -> None: ...',
            'add': ['def add(self, coeff: Complex, indices: List[int], ops: List[str]) -> None: ...', 
                   'def add(self, other: SQOperator) -> None: ...'],
            'add_term': 'def add_term(self, coeff: Complex, indices: List[int], ops: List[str]) -> None: ...',
            'add_op': 'def add_op(self, other: SQOperator) -> None: ...',
            'set_coeffs': 'def set_coeffs(self, coeffs: List[Complex]) -> None: ...',
            'mult_coeffs': 'def mult_coeffs(self, factor: Complex) -> None: ...',
            'terms': 'def terms(self) -> List[Tuple[Complex, List[int], List[str]]]: ...',
            'get_largest_alfa_beta_indices': 'def get_largest_alfa_beta_indices(self) -> Tuple[int, int]: ...',
            'many_body_order': 'def many_body_order(self) -> int: ...',
            'ranks_present': 'def ranks_present(self) -> List[int]: ...',
            'canonical_order': 'def canonical_order(self) -> None: ...',
            'simplify': 'def simplify(self) -> None: ...',
            'jw_transform': 'def jw_transform(self, qubit_excitation: bool = False) -> QubitOperator: ...',
            'split_by_rank': 'def split_by_rank(self) -> Dict[int, SQOperator]: ...',
            'str': 'def str(self) -> str: ...',
            '__str__': 'def __str__(self) -> str: ...',
            '__repr__': 'def __repr__(self) -> str: ...'
        },
        # Add more class signatures as needed...
    }
    
    for class_name, class_info in classes.items():
        pyi_content += f'class {class_name}:\n'
        pyi_content += f'    """Python binding for {class_info["cpp_class"]}."""\n'
        
        # Add methods
        if class_name in class_signatures:
            signatures = class_signatures[class_name]
            for method in class_info['methods']:
                if method in signatures:
                    sig = signatures[method]
                    if isinstance(sig, list):
                        for s in sig:
                            pyi_content += f'    {s}\n'
                    else:
                        pyi_content += f'    {sig}\n'
                else:
                    # Generic signature for unknown methods
                    pyi_content += f'    def {method}(self, *args: Any, **kwargs: Any) -> Any: ...\n'
        else:
            # Generic class with basic methods
            pyi_content += '    def __init__(self, *args: Any, **kwargs: Any) -> None: ...\n'
            for method in class_info['methods']:
                if not method.startswith('__'):
                    pyi_content += f'    def {method}(self, *args: Any, **kwargs: Any) -> Any: ...\n'
        
        # Add static methods
        for static_method in class_info['static_methods']:
            pyi_content += f'    @staticmethod\n'
            pyi_content += f'    def {static_method}(*args: Any, **kwargs: Any) -> Any: ...\n'
        
        pyi_content += '\n'
    
    # Add standalone functions
    if functions:
        pyi_content += '# Standalone functions\n'
        for func in functions:
            if func == 'gate':
                # Special handling for gate function overloads
                pyi_content += '''@overload
def gate(type: str, target: int, parameter: Complex = 0.0) -> Gate: ...

@overload
def gate(type: str, target: int, control: int) -> Gate: ...

@overload
def gate(type: str, target: int, control: int, parameter: Complex = 0.0) -> Gate: ...

'''
            elif func == 'control_gate':
                pyi_content += 'def control_gate(control: int, gate: Gate) -> Gate: ...\n'
            else:
                pyi_content += f'def {func}(*args: Any, **kwargs: Any) -> Any: ...\n'
    
    return pyi_content


def main():
    """Main function to generate the .pyi file."""
    
    # Get the script directory and project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Paths
    bindings_path = project_root / "src" / "qforte" / "bindings.cc"
    pyi_path = project_root / "src" / "qforte" / "qforte.pyi"
    
    if not bindings_path.exists():
        print(f"Error: bindings.cc not found at {bindings_path}")
        sys.exit(1)
    
    print(f"Parsing bindings from: {bindings_path}")
    classes, functions = parse_bindings_file(bindings_path)
    
    print(f"Found {len(classes)} classes and {len(functions)} functions")
    
    # Generate .pyi content
    pyi_content = generate_pyi_content(classes, functions)
    
    # Write to file
    with open(pyi_path, 'w') as f:
        f.write(pyi_content)
    
    print(f"Generated type stubs at: {pyi_path}")


if __name__ == "__main__":
    main()
