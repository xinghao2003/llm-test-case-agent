"""
User story parser to extract entities and requirements
"""
import re
from typing import Dict, List, Set


class UserStoryParser:
    """
    Parses user stories to extract structured information
    """

    def __init__(self):
        """Initialize the parser"""
        pass

    def parse(self, user_story: str) -> Dict:
        """
        Parse a user story into structured format

        Args:
            user_story: Natural language user story

        Returns:
            Dict with parsed information
        """
        return {
            'role': self._extract_role(user_story),
            'action': self._extract_action(user_story),
            'benefit': self._extract_benefit(user_story),
            'entities': self._extract_entities(user_story),
            'actions': self._extract_actions(user_story),
            'keywords': self._extract_keywords(user_story)
        }

    def _extract_role(self, story: str) -> str:
        """Extract user role from 'As a...' pattern"""
        match = re.search(r'As an?\s+([^,]+)', story, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return 'user'

    def _extract_action(self, story: str) -> str:
        """Extract main action from 'I want to...' pattern"""
        match = re.search(r'I want to\s+([^,\n]+)', story, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ''

    def _extract_benefit(self, story: str) -> str:
        """Extract benefit from 'so that...' pattern"""
        match = re.search(r'so that\s+([^.\n]+)', story, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ''

    def _extract_entities(self, story: str) -> List[str]:
        """Extract likely entity names (nouns)"""
        # Simple heuristic: capitalize words might be entities
        words = re.findall(r'\b[A-Z][a-z]+\b', story)
        # Also look for quoted entities
        quoted = re.findall(r'["\']([^"\']+)["\']', story)
        return list(set(words + quoted))

    def _extract_actions(self, story: str) -> List[str]:
        """Extract action verbs"""
        action_keywords = [
            'create', 'read', 'update', 'delete', 'register', 'login', 'logout',
            'upload', 'download', 'search', 'filter', 'sort', 'validate',
            'authenticate', 'authorize', 'send', 'receive', 'process'
        ]

        found_actions = []
        story_lower = story.lower()
        for keyword in action_keywords:
            if keyword in story_lower:
                found_actions.append(keyword)

        return found_actions

    def _extract_keywords(self, story: str) -> List[str]:
        """Extract important keywords"""
        # Common test-relevant keywords
        keywords = [
            'email', 'password', 'username', 'user', 'admin', 'account',
            'file', 'data', 'database', 'api', 'request', 'response',
            'valid', 'invalid', 'error', 'success', 'failure'
        ]

        found = []
        story_lower = story.lower()
        for keyword in keywords:
            if keyword in story_lower:
                found.append(keyword)

        return found
