"""
Code parsing and analysis utilities
"""
import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Set
import inspect


class PythonCodeParser:
    """
    Parses Python code files to extract structure and signatures
    """

    def __init__(self):
        """Initialize the parser"""
        pass

    def parse_file(self, file_path: str) -> Dict:
        """
        Parse a Python file and extract its structure

        Args:
            file_path: Path to Python file

        Returns:
            Dict with file structure information
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)

            return {
                'file_path': file_path,
                'imports': self._extract_imports(tree),
                'functions': self._extract_functions(tree),
                'classes': self._extract_classes(tree),
                'constants': self._extract_constants(tree),
                'docstring': ast.get_docstring(tree),
                'raw_content': content
            }
        except Exception as e:
            return {
                'file_path': file_path,
                'error': str(e),
                'imports': [],
                'functions': [],
                'classes': []
            }

    def _extract_imports(self, tree: ast.AST) -> List[Dict]:
        """Extract import statements"""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'type': 'import',
                        'module': alias.name,
                        'alias': alias.asname,
                        'statement': f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else "")
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append({
                        'type': 'from_import',
                        'module': module,
                        'name': alias.name,
                        'alias': alias.asname,
                        'statement': f"from {module} import {alias.name}" + (f" as {alias.asname}" if alias.asname else "")
                    })

        return imports

    def _extract_functions(self, tree: ast.AST) -> List[Dict]:
        """Extract function definitions"""
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip nested functions and class methods for now
                if self._is_top_level_function(node, tree):
                    func_info = {
                        'name': node.name,
                        'args': self._extract_function_args(node),
                        'returns': self._extract_return_annotation(node),
                        'docstring': ast.get_docstring(node),
                        'decorators': [self._get_decorator_name(dec) for dec in node.decorator_list],
                        'is_async': isinstance(node, ast.AsyncFunctionDef),
                        'line_number': node.lineno,
                        'signature': self._build_function_signature(node)
                    }
                    functions.append(func_info)

        return functions

    def _extract_classes(self, tree: ast.AST) -> List[Dict]:
        """Extract class definitions"""
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'bases': [self._get_name(base) for base in node.bases],
                    'methods': self._extract_class_methods(node),
                    'docstring': ast.get_docstring(node),
                    'decorators': [self._get_decorator_name(dec) for dec in node.decorator_list],
                    'line_number': node.lineno
                }
                classes.append(class_info)

        return classes

    def _extract_constants(self, tree: ast.AST) -> List[Dict]:
        """Extract module-level constants"""
        constants = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants.append({
                            'name': target.id,
                            'value': self._get_constant_value(node.value),
                            'line_number': node.lineno
                        })

        return constants

    def _extract_function_args(self, node: ast.FunctionDef) -> List[Dict]:
        """Extract function arguments with annotations"""
        args = []

        # Regular arguments
        for arg in node.args.args:
            args.append({
                'name': arg.arg,
                'annotation': ast.unparse(arg.annotation) if arg.annotation else None,
                'kind': 'positional'
            })

        # *args
        if node.args.vararg:
            args.append({
                'name': node.args.vararg.arg,
                'annotation': ast.unparse(node.args.vararg.annotation) if node.args.vararg.annotation else None,
                'kind': 'vararg'
            })

        # **kwargs
        if node.args.kwarg:
            args.append({
                'name': node.args.kwarg.arg,
                'annotation': ast.unparse(node.args.kwarg.annotation) if node.args.kwarg.annotation else None,
                'kind': 'kwarg'
            })

        return args

    def _extract_return_annotation(self, node: ast.FunctionDef) -> Optional[str]:
        """Extract return type annotation"""
        if node.returns:
            return ast.unparse(node.returns)
        return None

    def _extract_class_methods(self, class_node: ast.ClassDef) -> List[Dict]:
        """Extract methods from a class"""
        methods = []

        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                method_info = {
                    'name': node.name,
                    'args': self._extract_function_args(node),
                    'returns': self._extract_return_annotation(node),
                    'docstring': ast.get_docstring(node),
                    'is_static': any(self._get_decorator_name(dec) == 'staticmethod' for dec in node.decorator_list),
                    'is_class_method': any(self._get_decorator_name(dec) == 'classmethod' for dec in node.decorator_list),
                    'is_property': any(self._get_decorator_name(dec) == 'property' for dec in node.decorator_list),
                    'signature': self._build_function_signature(node)
                }
                methods.append(method_info)

        return methods

    def _build_function_signature(self, node: ast.FunctionDef) -> str:
        """Build a complete function signature string"""
        args = []

        # Regular args
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)

        # *args
        if node.args.vararg:
            arg_str = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                arg_str += f": {ast.unparse(node.args.vararg.annotation)}"
            args.append(arg_str)

        # **kwargs
        if node.args.kwarg:
            arg_str = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                arg_str += f": {ast.unparse(node.args.kwarg.annotation)}"
            args.append(arg_str)

        signature = f"def {node.name}({', '.join(args)})"

        if node.returns:
            signature += f" -> {ast.unparse(node.returns)}"

        return signature

    def _is_top_level_function(self, node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if function is at module level (not nested)"""
        # Simple heuristic: check if parent is Module
        for parent in ast.walk(tree):
            if isinstance(parent, ast.Module):
                if node in parent.body:
                    return True
        return False

    def _get_decorator_name(self, decorator) -> str:
        """Get decorator name as string"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
        return ast.unparse(decorator)

    def _get_name(self, node) -> str:
        """Get name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        return ast.unparse(node)

    def _get_constant_value(self, node) -> str:
        """Get constant value as string"""
        try:
            return ast.unparse(node)
        except:
            return "..."

    def extract_module_path(self, file_path: str, project_root: str) -> str:
        """
        Convert file path to Python module path

        Args:
            file_path: Absolute file path
            project_root: Project root directory

        Returns:
            Module path (e.g., 'src.auth.login')
        """
        file_path = Path(file_path)
        project_root = Path(project_root)

        # Get relative path
        try:
            rel_path = file_path.relative_to(project_root)
        except ValueError:
            rel_path = file_path

        # Convert to module path
        module_parts = list(rel_path.parts[:-1])  # Remove filename
        module_parts.append(rel_path.stem)  # Add filename without .py

        return '.'.join(module_parts)
