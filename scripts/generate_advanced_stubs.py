#!/usr/bin/env python3
"""
Advanced Python stub generator for qforte
This version generates more accurate type signatures by analyzing the C++ bindings
"""

import re
import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any


class StubGenerator:
    """Generate Python type stubs from pybind11 bindings"""
    
    def __init__(self, bindings_path: Path, output_path: Path):
        self.bindings_path = bindings_path
        self.output_path = output_path
        self.cpp_to_python_types = {
            'std::string': 'str',
            'string': 'str',
            'bool': 'bool',
            'int': 'int',
            'size_t': 'int',
            'float': 'float',
            'double': 'float',
            'std::complex<double>': 'Complex',
            'complex<double>': 'Complex',
            'void': 'None',
            'std::vector<int>': 'List[int]',
            'std::vector<size_t>': 'List[int]',
            'std::vector<std::string>': 'List[str]',
            'std::vector<std::complex<double>>': 'List[Complex]',
            'std::map<int, int>': 'Dict[int, int]',
            'std::map<std::tuple<int, int>, std::vector<std::tuple<int, float>>>': 'Dict[Tuple[int, int], List[Tuple[int, float]]]',
        }
    
    def parse_bindings(self) -> Dict[str, Any]:
        """Parse the bindings file and extract all relevant information"""
        with open(self.bindings_path, 'r') as f:
            content = f.read()
        
        classes = {}
        functions = []
        
        # Extract py::class_ definitions
        class_pattern = r'py::class_<([^>]+)>\(m,\s*"([^"]+)"\)(.*?)(?=py::class_|py::enum_|m\.def\(|$)'
        
        for match in re.finditer(class_pattern, content, re.DOTALL):
            cpp_class_spec = match.group(1).strip()
            python_name = match.group(2).strip()
            class_body = match.group(3)
            
            # Extract the main C++ class name
            cpp_class = cpp_class_spec.split(',')[0].strip()
            
            classes[python_name] = self._parse_class_body(cpp_class, class_body)
        
        # Extract standalone functions
        func_pattern = r'm\.def\(\s*"([^"]+)".*?\)'
        for func_match in re.finditer(func_pattern, content, re.DOTALL):
            functions.append(func_match.group(1))
        
        return {'classes': classes, 'functions': functions}
    
    def _parse_class_body(self, cpp_class: str, class_body: str) -> Dict[str, Any]:
        """Parse a single class body and extract methods"""
        class_info = {
            'cpp_class': cpp_class,
            'constructors': [],
            'methods': {},
            'static_methods': {},
            'properties': {}
        }
        
        # Parse constructors
        init_pattern = r'\.def\(py::init<([^>]*)>\(\)'
        for init_match in re.finditer(init_pattern, class_body):
            params = init_match.group(1).strip()
            class_info['constructors'].append(self._parse_parameters(params))
        
        # Parse regular methods
        method_pattern = r'\.def\(\s*"([^"]+)",\s*&[^:]+::([^,)]+)(?:,\s*([^)]+))?\)'
        for method_match in re.finditer(method_pattern, class_body):
            method_name = method_match.group(1)
            cpp_method = method_match.group(2)
            args_part = method_match.group(3) or ""
            
            # Extract arguments and default values
            method_info = self._parse_method_args(args_part)
            method_info['cpp_method'] = cpp_method
            
            if method_name not in class_info['methods']:
                class_info['methods'][method_name] = []
            class_info['methods'][method_name].append(method_info)
        
        # Parse static methods
        static_pattern = r'\.def_static\(\s*"([^"]+)",\s*&[^:]+::([^,)]+)'
        for static_match in re.finditer(static_pattern, class_body):
            method_name = static_match.group(1)
            cpp_method = static_match.group(2)
            class_info['static_methods'][method_name] = {'cpp_method': cpp_method}
        
        return class_info
    
    def _parse_parameters(self, params_str: str) -> List[str]:
        """Parse parameter string into list of types"""
        if not params_str.strip():
            return []
        
        # Simple split by comma, but this could be improved for complex types
        params = [p.strip() for p in params_str.split(',')]
        return [self._cpp_to_python_type(p) for p in params]
    
    def _parse_method_args(self, args_str: str) -> Dict[str, Any]:
        """Parse method arguments from pybind11 definition"""
        method_info = {
            'args': [],
            'defaults': {},
            'return_type': 'Any'
        }
        
        if not args_str:
            return method_info
        
        # Extract py::arg() patterns
        arg_pattern = r'py::arg\(\s*"([^"]+)"\s*\)(?:\s*=\s*([^,)]+))?'
        for arg_match in re.finditer(arg_pattern, args_str):
            arg_name = arg_match.group(1)
            default_value = arg_match.group(2)
            
            method_info['args'].append(arg_name)
            if default_value:
                method_info['defaults'][arg_name] = default_value.strip()
        
        return method_info
    
    def _cpp_to_python_type(self, cpp_type: str) -> str:
        """Convert C++ type to Python type annotation"""
        cpp_type = cpp_type.strip()
        
        # Handle const and reference types
        cpp_type = re.sub(r'\b(const|&)\b', '', cpp_type).strip()
        
        # Direct mappings
        if cpp_type in self.cpp_to_python_types:
            return self.cpp_to_python_types[cpp_type]
        
        # Handle template types
        if 'std::vector' in cpp_type:
            inner_match = re.search(r'std::vector<(.+)>', cpp_type)
            if inner_match:
                inner_type = self._cpp_to_python_type(inner_match.group(1))
                return f'List[{inner_type}]'
        
        # Handle custom classes (assume they exist in the module)
        if cpp_type in ['Circuit', 'Gate', 'SQOperator', 'QubitOperator', 'Tensor', 'Computer', 
                       'FCIComputer', 'FCIComputerGPU', 'FCIComputerThrust', 'TensorGPU', 
                       'TensorThrust', 'QubitBasis', 'SparseMatrix', 'SparseVector', 'local_timer']:
            return cpp_type
        
        # Default to Any for unknown types
        return 'Any'
    
    def generate_stub(self) -> str:
        """Generate the complete stub file content"""
        parsed = self.parse_bindings()
        
        stub_content = '''# qforte.pyi - Type stubs for the qforte quantum chemistry library
# Auto-generated from bindings.cc - DO NOT EDIT MANUALLY
# To regenerate: python scripts/generate_advanced_stubs.py

from typing import List, Dict, Tuple, Union, Optional, Iterator, Any, overload
import numpy as np
from numpy.typing import NDArray

# Complex number type alias
Complex = Union[complex, float]

'''
        
        # Generate classes
        for class_name, class_info in parsed['classes'].items():
            stub_content += self._generate_class_stub(class_name, class_info)
            stub_content += '\n'
        
        # Generate functions
        if parsed['functions']:
            stub_content += '# Standalone functions\n'
            for func_name in parsed['functions']:
                if func_name == 'gate':
                    stub_content += '''@overload
def gate(type: str, target: int, parameter: Complex = 0.0) -> Gate: ...

@overload
def gate(type: str, target: int, control: int) -> Gate: ...

@overload
def gate(type: str, target: int, control: int, parameter: Complex = 0.0) -> Gate: ...

'''
                elif func_name == 'control_gate':
                    stub_content += 'def control_gate(control: int, gate: Gate) -> Gate: ...\n'
                else:
                    stub_content += f'def {func_name}(*args: Any, **kwargs: Any) -> Any: ...\n'
        
        return stub_content
    
    def _generate_class_stub(self, class_name: str, class_info: Dict[str, Any]) -> str:
        """Generate stub for a single class"""
        lines = [f'class {class_name}:']
        lines.append(f'    """Python binding for {class_info["cpp_class"]}."""')
        
        # Constructors
        if class_info['constructors']:
            for i, constructor in enumerate(class_info['constructors']):
                if not constructor:  # No-argument constructor
                    lines.append('    def __init__(self) -> None: ...')
                else:
                    params = ', '.join(f'arg{j}: {param}' for j, param in enumerate(constructor))
                    lines.append(f'    def __init__(self, {params}) -> None: ...')
        else:
            # Default constructor if none specified
            lines.append('    def __init__(self, *args: Any, **kwargs: Any) -> None: ...')
        
        # Regular methods
        for method_name, method_overloads in class_info['methods'].items():
            if method_name.startswith('__'):
                continue  # Skip magic methods for now
                
            for method_info in method_overloads:
                signature = self._generate_method_signature(method_name, method_info)
                lines.append(f'    {signature}')
        
        # Static methods
        for method_name, method_info in class_info['static_methods'].items():
            lines.append('    @staticmethod')
            signature = self._generate_method_signature(method_name, method_info, is_static=True)
            lines.append(f'    {signature}')
        
        return '\n'.join(lines)
    
    def _generate_method_signature(self, method_name: str, method_info: Dict[str, Any], 
                                 is_static: bool = False) -> str:
        """Generate method signature string"""
        if is_static:
            params = []
        else:
            params = ['self']
        
        # Add arguments with types and defaults
        for arg in method_info.get('args', []):
            param_str = f'{arg}: Any'
            if arg in method_info.get('defaults', {}):
                default = method_info['defaults'][arg]
                param_str += f' = {default}'
            params.append(param_str)
        
        params_str = ', '.join(params)
        return_type = method_info.get('return_type', 'Any')
        
        return f'def {method_name}({params_str}) -> {return_type}: ...'


def main():
    """Main function to generate advanced stubs"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    bindings_path = project_root / "src" / "qforte" / "bindings.cc"
    output_path = project_root / "src" / "qforte" / "qforte_generated.pyi"
    
    if not bindings_path.exists():
        print(f"Error: {bindings_path} not found")
        return
    
    generator = StubGenerator(bindings_path, output_path)
    stub_content = generator.generate_stub()
    
    with open(output_path, 'w') as f:
        f.write(stub_content)
    
    print(f"Generated advanced type stubs at: {output_path}")
    print("Review the generated file and merge with your existing .pyi file as needed.")


if __name__ == "__main__":
    main()
