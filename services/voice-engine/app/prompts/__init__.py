"""
Prompt Loader Module

Loads and manages recording prompts for voice model training.
Supports English (en) and Icelandic (is) languages.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Directory containing prompt files
PROMPTS_DIR = Path(__file__).parent


@dataclass
class PromptCategory:
    """A category of prompts"""
    name: str
    description: str
    phonemes_covered: List[str] = field(default_factory=list)
    prompts: List[str] = field(default_factory=list)


@dataclass
class LanguagePrompts:
    """All prompts for a language"""
    language: str
    language_name: str
    total_prompts: int
    description: str
    categories: Dict[str, PromptCategory]
    phoneme_coverage_map: Dict[str, List[str]]  # phoneme -> categories
    
    def get_all_prompts(self) -> List[str]:
        """Get all prompts across all categories"""
        all_prompts = []
        for category in self.categories.values():
            all_prompts.extend(category.prompts)
        return all_prompts
    
    def get_prompts_by_category(self, category_name: str) -> List[str]:
        """Get prompts for a specific category"""
        if category_name in self.categories:
            return self.categories[category_name].prompts
        return []
    
    def get_prompts_for_phonemes(self, phonemes: List[str]) -> List[str]:
        """Get prompts that cover specific phonemes"""
        prompts = set()
        for phoneme in phonemes:
            if phoneme in self.phoneme_coverage_map:
                for category_name in self.phoneme_coverage_map[phoneme]:
                    if category_name in self.categories:
                        for prompt in self.categories[category_name].prompts:
                            prompts.add(prompt)
        return list(prompts)
    
    def get_random_prompts(
        self, 
        count: int, 
        exclude_categories: Optional[List[str]] = None
    ) -> List[str]:
        """Get random prompts"""
        all_prompts = []
        for name, category in self.categories.items():
            if exclude_categories and name in exclude_categories:
                continue
            all_prompts.extend(category.prompts)
        
        if count >= len(all_prompts):
            random.shuffle(all_prompts)
            return all_prompts
        
        return random.sample(all_prompts, count)
    
    def get_balanced_prompt_set(self, target_count: int = 50) -> List[str]:
        """
        Get a balanced set of prompts covering all categories.
        Distributes prompts evenly across categories.
        """
        prompts = []
        category_count = len(self.categories)
        prompts_per_category = max(1, target_count // category_count)
        
        for category in self.categories.values():
            category_prompts = category.prompts[:prompts_per_category]
            prompts.extend(category_prompts)
        
        # Shuffle for variety
        random.shuffle(prompts)
        return prompts[:target_count]


class PromptLoader:
    """
    Loads and manages prompts for different languages.
    
    Usage:
        loader = PromptLoader()
        en_prompts = loader.get_language("en")
        all_prompts = en_prompts.get_all_prompts()
    """
    
    def __init__(self, prompts_dir: Optional[Path] = None):
        self.prompts_dir = prompts_dir or PROMPTS_DIR
        self._cache: Dict[str, LanguagePrompts] = {}
    
    def get_available_languages(self) -> List[str]:
        """Get list of available language codes"""
        languages = []
        for file in self.prompts_dir.glob("*_prompts.json"):
            lang_code = file.stem.replace("_prompts", "")
            languages.append(lang_code)
        return sorted(languages)
    
    def get_language(self, language: str) -> Optional[LanguagePrompts]:
        """
        Load prompts for a specific language.
        
        Args:
            language: Language code (en, is)
            
        Returns:
            LanguagePrompts or None if not found
        """
        if language in self._cache:
            return self._cache[language]
        
        prompt_file = self.prompts_dir / f"{language}_prompts.json"
        
        if not prompt_file.exists():
            logger.warning(f"No prompts found for language: {language}")
            return None
        
        try:
            with open(prompt_file) as f:
                data = json.load(f)
            
            # Parse categories
            categories = {}
            for name, cat_data in data.get("categories", {}).items():
                categories[name] = PromptCategory(
                    name=name,
                    description=cat_data.get("description", ""),
                    phonemes_covered=cat_data.get("phonemes_covered", []),
                    prompts=cat_data.get("prompts", [])
                )
            
            language_prompts = LanguagePrompts(
                language=data.get("language", language),
                language_name=data.get("language_name", language),
                total_prompts=data.get("total_prompts", 0),
                description=data.get("description", ""),
                categories=categories,
                phoneme_coverage_map=data.get("phoneme_coverage_map", {})
            )
            
            self._cache[language] = language_prompts
            return language_prompts
            
        except Exception as e:
            logger.error(f"Error loading prompts for {language}: {e}")
            return None
    
    def get_prompts_for_missing_phonemes(
        self,
        language: str,
        missing_phonemes: Set[str],
        max_prompts: int = 20
    ) -> List[str]:
        """
        Get prompts specifically for covering missing phonemes.
        
        Args:
            language: Language code
            missing_phonemes: Set of missing phonemes to cover
            max_prompts: Maximum number of prompts to return
            
        Returns:
            List of prompts that cover the missing phonemes
        """
        lang_prompts = self.get_language(language)
        if not lang_prompts:
            return []
        
        prompts = lang_prompts.get_prompts_for_phonemes(list(missing_phonemes))
        
        if len(prompts) > max_prompts:
            # Prioritize prompts that cover multiple missing phonemes
            # For now, just shuffle and take first N
            random.shuffle(prompts)
            return prompts[:max_prompts]
        
        return prompts
    
    def get_wizard_session_prompts(
        self,
        language: str,
        session_length: int = 50,
        target_phonemes: Optional[Set[str]] = None
    ) -> List[Dict]:
        """
        Get prompts for a recording wizard session.
        
        Args:
            language: Language code
            session_length: Number of prompts for the session
            target_phonemes: Optional specific phonemes to target
            
        Returns:
            List of prompt dicts with metadata
        """
        lang_prompts = self.get_language(language)
        if not lang_prompts:
            return []
        
        if target_phonemes:
            prompts = lang_prompts.get_prompts_for_phonemes(list(target_phonemes))
        else:
            prompts = lang_prompts.get_balanced_prompt_set(session_length)
        
        # Convert to list of dicts with metadata
        result = []
        for i, prompt in enumerate(prompts[:session_length]):
            # Find which category this prompt belongs to
            category = "general"
            for cat_name, cat in lang_prompts.categories.items():
                if prompt in cat.prompts:
                    category = cat_name
                    break
            
            result.append({
                "index": i + 1,
                "total": session_length,
                "prompt": prompt,
                "category": category,
                "language": language
            })
        
        return result


# Singleton instance
_loader: Optional[PromptLoader] = None


def get_prompt_loader() -> PromptLoader:
    """Get the singleton prompt loader instance"""
    global _loader
    if _loader is None:
        _loader = PromptLoader()
    return _loader


# Convenience functions
def get_prompts(language: str) -> Optional[LanguagePrompts]:
    """Get prompts for a language"""
    return get_prompt_loader().get_language(language)


def get_available_languages() -> List[str]:
    """Get list of available languages"""
    return get_prompt_loader().get_available_languages()


def get_wizard_prompts(
    language: str, 
    count: int = 50,
    target_phonemes: Optional[Set[str]] = None
) -> List[Dict]:
    """Get prompts for wizard session"""
    return get_prompt_loader().get_wizard_session_prompts(
        language, count, target_phonemes
    )


if __name__ == "__main__":
    # Test the loader
    loader = PromptLoader()
    
    print(f"Available languages: {loader.get_available_languages()}")
    
    for lang in ["en", "is"]:
        prompts = loader.get_language(lang)
        if prompts:
            print(f"\n{prompts.language_name}:")
            print(f"  Total prompts: {prompts.total_prompts}")
            print(f"  Categories: {list(prompts.categories.keys())}")
            print(f"  Sample prompts:")
            for p in prompts.get_random_prompts(3):
                print(f"    - {p}")
