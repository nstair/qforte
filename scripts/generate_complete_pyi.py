#!/usr/bin/env python3
"""
Automatic .pyi stub file generator for qforte
This script parses the pybind11 bindings and generates complete type stubs.
"""

import re
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union


class TypeSignatureGenerator:
    """Generates appropriate type signatures for different C++ types."""
    
    TYPE_MAP = {
        'int': 'int',
        'size_t': 'int',
        'double': 'float',
        'float': 'float',
        'bool': 'bool',
        'std::string': 'str',
        'std::complex<double>': 'Complex',
        'std::vector<int>': 'List[int]',
        'std::vector<size_t>': 'List[int]',
        'std::vector<double>': 'List[float]',
        'std::vector<std::string>': 'List[str]',
        'std::vector<std::complex<double>>': 'List[Complex]',
        'std::map<int, int>': 'Dict[int, int]',
        'std::unordered_map<int, int>': 'Dict[int, int]',
    }
    
    @classmethod
    def map_type(cls, cpp_type: str) -> str:
        """Map C++ type to Python type annotation."""
        cpp_type = cpp_type.strip()
        
        # Direct mapping
        if cpp_type in cls.TYPE_MAP:
            return cls.TYPE_MAP[cpp_type]
        
        # Handle template types
        if 'std::vector<' in cpp_type:
            inner_type = cpp_type[cpp_type.find('<')+1:cpp_type.rfind('>')]
            mapped_inner = cls.map_type(inner_type)
            return f'List[{mapped_inner}]'
        
        if 'std::map<' in cpp_type or 'std::unordered_map<' in cpp_type:
            start = cpp_type.find('<') + 1
            end = cpp_type.rfind('>')
            inner = cpp_type[start:end]
            types = [t.strip() for t in inner.split(',')]
            if len(types) == 2:
                key_type = cls.map_type(types[0])
                value_type = cls.map_type(types[1])
                return f'Dict[{key_type}, {value_type}]'
        
        # Handle qforte types
        if cpp_type in ['Circuit', 'Gate', 'SQOperator', 'QubitOperator', 'Tensor', 
                       'TensorGPU', 'TensorThrust', 'FCIComputer', 'FCIComputerGPU', 
                       'FCIComputerThrust', 'QubitBasis', 'Computer', 'SparseMatrix',
                       'SparseVector', 'local_timer', 'TensorOperator', 'SQOpPool',
                       'QubitOpPool', 'FCIGraph']:
            return cpp_type
        
        # Default to Any for unknown types
        return 'Any'


class BindingsParser:
    """Parser for pybind11 bindings.cc file."""
    
    def __init__(self, bindings_path: Path):
        self.bindings_path = bindings_path
        self.type_gen = TypeSignatureGenerator()
    
    def parse(self) -> Tuple[Dict, List]:
        """Parse the bindings file and extract class and function information."""
        with open(self.bindings_path, 'r') as f:
            content = f.read()
        
        classes = self._parse_classes(content)
        functions = self._parse_functions(content)
        
        return classes, functions
    
    def _parse_classes(self, content: str) -> Dict:
        """Parse class definitions from bindings."""
        classes = {}
        
        # Pattern to match py::class_ definitions
        class_pattern = r'py::class_<([^>]+)>\(m,\s*"([^"]+)"\)(.*?)(?=(?:py::class_|py::enum_|m\.def|$))'
        
        for match in re.finditer(class_pattern, content, re.DOTALL):
            cpp_types = match.group(1).strip()
            python_name = match.group(2).strip()
            class_body = match.group(3)
            
            # Extract the primary C++ class name
            cpp_class = cpp_types.split(',')[0].strip()
            
            # Parse constructors
            constructors = self._parse_constructors(class_body)
            
            # Parse methods
            methods = self._parse_methods(class_body)
            
            # Parse static methods
            static_methods = self._parse_static_methods(class_body)
            
            # Parse special methods (__iter__, __getitem__, etc.)
            special_methods = self._parse_special_methods(class_body)
            
            classes[python_name] = {
                'cpp_class': cpp_class,
                'constructors': constructors,
                'methods': methods,
                'static_methods': static_methods,
                'special_methods': special_methods
            }
        
        return classes
    
    def _parse_constructors(self, class_body: str) -> List[str]:
        """Parse constructor signatures."""
        constructors = []
        
        # Pattern for py::init<...>()
        init_pattern = r'\.def\(py::init<([^>]*)>\(\)'
        for match in re.finditer(init_pattern, class_body):
            params = match.group(1).strip()
            constructors.append(params)
        
        return constructors
    
    def _parse_methods(self, class_body: str) -> Dict[str, List]:
        """Parse method definitions."""
        methods = {}
        
        # Pattern for .def("method_name", ...)
        def_pattern = r'\.def\(\s*"([^"]+)"[^)]*\)'
        
        for match in re.finditer(def_pattern, class_body):
            method_name = match.group(1)
            if method_name not in methods:
                methods[method_name] = []
            
            # Try to extract parameter information from py::arg calls
            full_match = match.group(0)
            args = self._extract_args(full_match)
            methods[method_name].append(args)
        
        return methods
    
    def _parse_static_methods(self, class_body: str) -> Dict[str, str]:
        """Parse static method definitions."""
        static_methods = {}
        
        static_pattern = r'\.def_static\(\s*"([^"]+)"'
        for match in re.finditer(static_pattern, class_body):
            method_name = match.group(1)
            static_methods[method_name] = ""
        
        return static_methods
    
    def _parse_special_methods(self, class_body: str) -> Dict[str, str]:
        """Parse special Python methods like __iter__, __getitem__, etc."""
        special_methods = {}
        
        # Look for lambda definitions that implement special methods
        if '__iter__' in class_body:
            special_methods['__iter__'] = 'Iterator'
        if '__getitem__' in class_body:
            special_methods['__getitem__'] = 'getitem'
        if '__len__' in class_body:
            special_methods['__len__'] = 'len'
        
        return special_methods
    
    def _extract_args(self, def_statement: str) -> List[str]:
        """Extract argument information from a .def() statement."""
        args = []
        
        # Look for py::arg("name") patterns
        arg_pattern = r'py::arg\(\s*"([^"]+)"\)'
        for match in re.finditer(arg_pattern, def_statement):
            args.append(match.group(1))
        
        return args
    
    def _parse_functions(self, content: str) -> List[str]:
        """Parse standalone function definitions."""
        functions = []
        
        func_pattern = r'm\.def\(\s*"([^"]+)"'
        for match in re.finditer(func_pattern, content):
            functions.append(match.group(1))
        
        return functions


class StubGenerator:
    """Generates complete .pyi stub files."""
    
    def __init__(self):
        self.type_gen = TypeSignatureGenerator()
    
    def generate_stub_content(self, classes: Dict, functions: List) -> str:
        """Generate the complete .pyi file content."""
        
        content = '''# qforte.pyi - Type stubs for the qforte quantum chemistry library
# Auto-generated from bindings.cc - DO NOT EDIT MANUALLY
from typing import List, Dict, Tuple, Union, Optional, Iterator, Any, overload
import numpy as np
from numpy.typing import NDArray

# Complex number type alias
Complex = Union[complex, float]

'''
        
        # Generate class stubs
        for class_name, class_info in classes.items():
            content += self._generate_class_stub(class_name, class_info)
            content += '\n'
        
        # Generate function stubs
        if functions:
            content += '# Standalone functions\n'
            for func_name in functions:
                content += self._generate_function_stub(func_name)
            content += '\n'
        
        return content
    
    def _generate_class_stub(self, class_name: str, class_info: Dict) -> str:
        """Generate stub for a single class."""
        
        # Get class-specific signatures
        class_signatures = self._get_class_signatures(class_name)
        
        # Get docstring
        try:
            from class_signatures import CLASS_DOCSTRINGS
            docstring = CLASS_DOCSTRINGS.get(class_name, f'Python binding for {class_info["cpp_class"]}.')
        except ImportError:
            docstring = f'Python binding for {class_info["cpp_class"]}.'
        
        stub = f'class {class_name}:\n'
        stub += f'    """{docstring}"""\n'
        
        # Constructors
        try:
            from class_signatures import CONSTRUCTOR_SIGNATURES
            constructor_sigs = CONSTRUCTOR_SIGNATURES.get(class_name, [])
        except ImportError:
            constructor_sigs = []
        
        if isinstance(constructor_sigs, list):
            # Multiple constructors
            if constructor_sigs:
                for ctor_sig in constructor_sigs:
                    if ctor_sig:
                        stub += f'    def __init__(self, {ctor_sig}) -> None: ...\n'
                    else:
                        stub += '    def __init__(self) -> None: ...\n'
            else:
                stub += '    def __init__(self) -> None: ...\n'
        else:
            # Single constructor
            if constructor_sigs:
                stub += f'    def __init__(self, {constructor_sigs}) -> None: ...\n'
            else:
                stub += '    def __init__(self) -> None: ...\n'
        
        # Regular methods
        methods = class_info.get('methods', {})
        for method_name, method_variants in methods.items():
            if method_name in class_signatures:
                signatures = class_signatures[method_name]
                if isinstance(signatures, list):
                    for sig in signatures:
                        stub += f'    {sig}\n'
                else:
                    stub += f'    {signatures}\n'
            else:
                # Generate generic signature
                stub += f'    def {method_name}(self, *args: Any, **kwargs: Any) -> Any: ...\n'
        
        # Static methods
        static_methods = class_info.get('static_methods', {})
        for static_method in static_methods:
            if static_method in class_signatures:
                signature = class_signatures[static_method]
                if signature.startswith('@staticmethod'):
                    stub += f'    {signature}\n'
                else:
                    stub += f'    @staticmethod\n'
                    stub += f'    {signature}\n'
            else:
                stub += f'    @staticmethod\n'
                stub += f'    def {static_method}(*args: Any, **kwargs: Any) -> Any: ...\n'
        
        # Special methods
        special_methods = class_info.get('special_methods', {})
        for special_method, method_type in special_methods.items():
            if special_method in class_signatures:
                signature = class_signatures[special_method]
                stub += f'    {signature}\n'
            else:
                if special_method == '__iter__':
                    stub += f'    def __iter__(self) -> Iterator[Any]: ...\n'
                elif special_method == '__getitem__':
                    stub += f'    def __getitem__(self, key: Any) -> Any: ...\n'
                elif special_method == '__len__':
                    stub += f'    def __len__(self) -> int: ...\n'
        
        return stub
    
    def _parse_constructor_params(self, ctor_params: str, class_name: str) -> str:
        """Parse constructor parameters and generate appropriate type hints."""
        
        # Import the signature database
        try:
            from class_signatures import CONSTRUCTOR_SIGNATURES
            if class_name in CONSTRUCTOR_SIGNATURES:
                signature = CONSTRUCTOR_SIGNATURES[class_name]
                if isinstance(signature, list):
                    return signature[0] if signature else ""
                return signature
        except ImportError:
            pass
            
        if not ctor_params.strip():
            return ""
        
        # Generic parameter parsing fallback
        params = [p.strip() for p in ctor_params.split(',') if p.strip()]
        mapped_params = []
        for i, param in enumerate(params):
            param_type = self.type_gen.map_type(param)
            mapped_params.append(f'arg{i}: {param_type}')
        return ', '.join(mapped_params)
    
    def _generate_function_stub(self, func_name: str) -> str:
        """Generate stub for a standalone function."""
        
        if func_name == 'gate':
            return '''@overload
def gate(type: str, target: int, parameter: Complex = 0.0) -> Gate: ...

@overload
def gate(type: str, target: int, control: int) -> Gate: ...

@overload
def gate(type: str, target: int, control: int, parameter: Complex = 0.0) -> Gate: ...

'''
        elif func_name == 'control_gate':
            return 'def control_gate(control: int, gate: Gate) -> Gate: ...\n'
        else:
            return f'def {func_name}(*args: Any, **kwargs: Any) -> Any: ...\n'
    
    def _get_class_signatures(self, class_name: str) -> Dict[str, Union[str, List[str]]]:
        """Get predefined method signatures for known classes."""
        
        # Import the comprehensive signature database
        try:
            from class_signatures import CLASS_SIGNATURES
            return CLASS_SIGNATURES.get(class_name, {})
        except ImportError:
            # Fallback to minimal signatures if database not available
            return {}


def main():
    """Main function to generate the .pyi file."""
    
    # Get paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    bindings_path = project_root / "src" / "qforte" / "bindings.cc"
    pyi_path = project_root / "src" / "qforte" / "qforte.pyi"
    
    if not bindings_path.exists():
        print(f"Error: bindings.cc not found at {bindings_path}")
        sys.exit(1)
    
    print(f"Parsing bindings from: {bindings_path}")
    
    # Parse bindings
    parser = BindingsParser(bindings_path)
    classes, functions = parser.parse()
    
    print(f"Found {len(classes)} classes and {len(functions)} functions")
    
    # Generate stub content
    generator = StubGenerator()
    
    # Read the existing .pyi file to preserve manually written content
    existing_content = ""
    if pyi_path.exists():
        with open(pyi_path, 'r') as f:
            existing_content = f.read()
    
    # Generate new content but preserve manual additions like system_factory
    stub_content = generator.generate_stub_content(classes, functions)
    
    # Add any manual additions that might exist
    if 'system_factory' in existing_content and 'system_factory' not in stub_content:
        stub_content += '''
# Additional functions that may be imported from submodules
def system_factory(
    build_type: str,
    mol_geometry: List[Tuple[str, Tuple[float, float, float]]],
    basis: str,
    build_qb_ham: bool = True,
    run_fci: int = 0,
    **kwargs: Any
) -> Any: ...
'''
    
    # Write to file
    with open(pyi_path, 'w') as f:
        f.write(stub_content)
    
    print(f"Generated complete type stubs at: {pyi_path}")


if __name__ == "__main__":
    main()
