"""
Codebase analyzer to find relevant code for user stories
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Set
from .code_parser import PythonCodeParser
from .user_story_parser import UserStoryParser


class CodebaseAnalyzer:
    """
    Analyzes a codebase to find code relevant to a user story
    """

    def __init__(self):
        """Initialize the analyzer"""
        self.code_parser = PythonCodeParser()
        self.story_parser = UserStoryParser()

    def analyze_project(self, project_path: str, max_files: int = 50) -> Dict:
        """
        Analyze entire project structure

        Args:
            project_path: Path to project root
            max_files: Maximum number of files to analyze

        Returns:
            Dict with project structure
        """
        project_path = Path(project_path)

        if not project_path.exists():
            raise ValueError(f"Project path does not exist: {project_path}")

        python_files = self._find_python_files(project_path, max_files)

        project_info = {
            'project_path': str(project_path),
            'total_files': len(python_files),
            'files': [],
            'all_functions': [],
            'all_classes': []
        }

        for file_path in python_files:
            file_info = self.code_parser.parse_file(str(file_path))

            if 'error' not in file_info:
                # Add module path
                file_info['module_path'] = self.code_parser.extract_module_path(
                    str(file_path),
                    str(project_path)
                )

                project_info['files'].append(file_info)
                project_info['all_functions'].extend([
                    {**func, 'file': str(file_path), 'module': file_info['module_path']}
                    for func in file_info['functions']
                ])
                project_info['all_classes'].extend([
                    {**cls, 'file': str(file_path), 'module': file_info['module_path']}
                    for cls in file_info['classes']
                ])

        return project_info

    def find_relevant_code(
        self,
        user_story: str,
        project_path: str,
        max_files: int = 50
    ) -> Dict:
        """
        Find code files and functions relevant to a user story

        Args:
            user_story: Natural language user story
            project_path: Path to project root
            max_files: Maximum files to analyze

        Returns:
            Dict with relevant code information
        """
        # Parse user story
        story_info = self.story_parser.parse(user_story)

        # Analyze project
        project_info = self.analyze_project(project_path, max_files)

        # Find relevant files
        relevant_files = self._match_files_to_story(
            story_info,
            project_info['files']
        )

        # Find relevant functions
        relevant_functions = self._match_functions_to_story(
            story_info,
            project_info['all_functions']
        )

        # Find relevant classes
        relevant_classes = self._match_classes_to_story(
            story_info,
            project_info['all_classes']
        )

        return {
            'story_info': story_info,
            'project_path': project_path,
            'relevant_files': relevant_files,
            'relevant_functions': relevant_functions,
            'relevant_classes': relevant_classes,
            'suggested_test_location': self._suggest_test_location(
                project_path,
                relevant_files
            )
        }

    def _find_python_files(self, project_path: Path, max_files: int) -> List[Path]:
        """
        Find Python files in project, excluding common directories

        Args:
            project_path: Project root path
            max_files: Maximum number of files

        Returns:
            List of Python file paths
        """
        exclude_dirs = {
            '__pycache__', '.git', '.venv', 'venv', 'env',
            'node_modules', '.pytest_cache', '.mypy_cache',
            'htmlcov', 'dist', 'build', '.tox', 'eggs'
        }

        python_files = []

        for root, dirs, files in os.walk(project_path):
            # Remove excluded directories from search
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file.endswith('.py') and not file.startswith('test_'):
                    python_files.append(Path(root) / file)

                    if len(python_files) >= max_files:
                        return python_files

        return python_files

    def _match_files_to_story(
        self,
        story_info: Dict,
        files: List[Dict]
    ) -> List[Dict]:
        """
        Match files to user story based on keywords

        Args:
            story_info: Parsed user story information
            files: List of parsed file information

        Returns:
            List of relevant files with relevance scores
        """
        relevant = []
        keywords = story_info.get('keywords', []) + story_info.get('actions', [])

        for file_info in files:
            score = self._calculate_relevance_score(file_info, keywords)

            if score > 0:
                relevant.append({
                    **file_info,
                    'relevance_score': score
                })

        # Sort by relevance
        relevant.sort(key=lambda x: x['relevance_score'], reverse=True)

        return relevant[:10]  # Top 10 most relevant

    def _match_functions_to_story(
        self,
        story_info: Dict,
        functions: List[Dict]
    ) -> List[Dict]:
        """Match functions to user story"""
        relevant = []
        keywords = story_info.get('keywords', []) + story_info.get('actions', [])

        for func in functions:
            score = 0
            func_name_lower = func['name'].lower()

            # Check function name
            for keyword in keywords:
                if keyword.lower() in func_name_lower:
                    score += 10

            # Check docstring
            if func.get('docstring'):
                docstring_lower = func['docstring'].lower()
                for keyword in keywords:
                    if keyword.lower() in docstring_lower:
                        score += 5

            if score > 0:
                relevant.append({
                    **func,
                    'relevance_score': score
                })

        relevant.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant[:20]  # Top 20

    def _match_classes_to_story(
        self,
        story_info: Dict,
        classes: List[Dict]
    ) -> List[Dict]:
        """Match classes to user story"""
        relevant = []
        keywords = story_info.get('keywords', []) + story_info.get('actions', [])
        entities = story_info.get('entities', [])

        for cls in classes:
            score = 0
            class_name_lower = cls['name'].lower()

            # Check class name against keywords
            for keyword in keywords:
                if keyword.lower() in class_name_lower:
                    score += 10

            # Check against entities
            for entity in entities:
                if entity.lower() in class_name_lower:
                    score += 15

            # Check docstring
            if cls.get('docstring'):
                docstring_lower = cls['docstring'].lower()
                for keyword in keywords:
                    if keyword.lower() in docstring_lower:
                        score += 5

            if score > 0:
                relevant.append({
                    **cls,
                    'relevance_score': score
                })

        relevant.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant[:10]  # Top 10

    def _calculate_relevance_score(self, file_info: Dict, keywords: List[str]) -> int:
        """Calculate how relevant a file is to the keywords"""
        score = 0
        file_path_lower = file_info['file_path'].lower()

        # Check file path
        for keyword in keywords:
            if keyword.lower() in file_path_lower:
                score += 5

        # Check function names
        for func in file_info.get('functions', []):
            func_name_lower = func['name'].lower()
            for keyword in keywords:
                if keyword.lower() in func_name_lower:
                    score += 10

        # Check class names
        for cls in file_info.get('classes', []):
            class_name_lower = cls['name'].lower()
            for keyword in keywords:
                if keyword.lower() in class_name_lower:
                    score += 10

        # Check file docstring
        if file_info.get('docstring'):
            docstring_lower = file_info['docstring'].lower()
            for keyword in keywords:
                if keyword.lower() in docstring_lower:
                    score += 3

        return score

    def _suggest_test_location(
        self,
        project_path: str,
        relevant_files: List[Dict]
    ) -> str:
        """
        Suggest where to place test files

        Args:
            project_path: Project root
            relevant_files: Relevant source files

        Returns:
            Suggested test directory path
        """
        project_path = Path(project_path)

        # Check for existing test directories
        common_test_dirs = ['tests', 'test', 'testing']

        for test_dir in common_test_dirs:
            test_path = project_path / test_dir
            if test_path.exists() and test_path.is_dir():
                return str(test_path)

        # Create tests directory at project root
        return str(project_path / 'tests')

    def format_code_context_for_prompt(self, relevant_code: Dict) -> str:
        """
        Format relevant code information for LLM prompt

        Args:
            relevant_code: Output from find_relevant_code()

        Returns:
            Formatted string for prompt
        """
        lines = []

        # Relevant functions
        if relevant_code['relevant_functions']:
            lines.append("RELEVANT FUNCTIONS:")
            for func in relevant_code['relevant_functions'][:5]:  # Top 5
                lines.append(f"\nFile: {func['file']}")
                lines.append(f"Module: {func['module']}")
                lines.append(f"Signature: {func['signature']}")
                if func.get('docstring'):
                    lines.append(f"Description: {func['docstring']}")
                lines.append("")

        # Relevant classes
        if relevant_code['relevant_classes']:
            lines.append("RELEVANT CLASSES:")
            for cls in relevant_code['relevant_classes'][:3]:  # Top 3
                lines.append(f"\nFile: {cls['file']}")
                lines.append(f"Module: {cls['module']}")
                lines.append(f"Class: {cls['name']}")
                if cls.get('docstring'):
                    lines.append(f"Description: {cls['docstring']}")

                # Show key methods
                if cls.get('methods'):
                    lines.append("Key methods:")
                    for method in cls['methods'][:5]:
                        lines.append(f"  - {method['signature']}")
                lines.append("")

        return '\n'.join(lines)
