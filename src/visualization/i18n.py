"""
Internationalization (i18n) Module
Provides translation support for the dashboard.

Best Practices:
- Centralized translation management
- Easy to add new languages
- Simple API: t('key.path')
- Nested key support
- Fallback to English if translation missing
"""

import json
from pathlib import Path
from typing import Any, Dict
import streamlit as st


class TranslationManager:
    """Manages translations for multiple languages."""

    def __init__(self):
        self.locales_dir = Path(__file__).parent / "locales"
        self.translations: Dict[str, Dict] = {}
        self.available_languages = self._discover_languages()
        self.current_language = "kk"  # Default language (Kazakh)

    def _discover_languages(self) -> Dict[str, str]:
        """
        Discover available languages from locale files.

        Returns:
            Dictionary of language code to language name
        """
        languages = {}

        if not self.locales_dir.exists():
            return {"en": "English"}

        for file in self.locales_dir.glob("*.json"):
            lang_code = file.stem
            # Load the display name from the file
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    lang_name = data.get("_language_name", lang_code.upper())
                    languages[lang_code] = lang_name
            except:
                languages[lang_code] = lang_code.upper()

        return languages if languages else {"en": "English"}

    def load_language(self, lang_code: str) -> None:
        """
        Load translations for a specific language.

        Args:
            lang_code: Language code (e.g., 'en', 'kk', 'ru')
        """
        if lang_code in self.translations:
            self.current_language = lang_code
            return

        locale_file = self.locales_dir / f"{lang_code}.json"

        if not locale_file.exists():
            # Fallback to English
            locale_file = self.locales_dir / "en.json"
            if not locale_file.exists():
                # No translation files at all
                self.translations[lang_code] = {}
                self.current_language = lang_code
                return

        try:
            with open(locale_file, 'r', encoding='utf-8') as f:
                self.translations[lang_code] = json.load(f)
            self.current_language = lang_code
        except Exception as e:
            print(f"Warning: Could not load translations for {lang_code}: {e}")
            self.translations[lang_code] = {}
            self.current_language = lang_code

    def get_translation(self, key: str, **kwargs) -> str:
        """
        Get translation for a key with optional formatting.

        Supports nested keys using dot notation: 'section.subsection.key'

        Args:
            key: Translation key (e.g., 'pages.overview' or 'metrics.accuracy')
            **kwargs: Optional formatting arguments

        Returns:
            Translated string or key itself if not found
        """
        # Load current language if not loaded
        if self.current_language not in self.translations:
            self.load_language(self.current_language)

        # Navigate nested keys
        keys = key.split('.')
        value = self.translations.get(self.current_language, {})

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                break

        # If translation not found, try English fallback
        if value is None and self.current_language != 'en':
            if 'en' not in self.translations:
                self.load_language('en')

            value = self.translations.get('en', {})
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k)
                else:
                    break

        # If still not found, return the key itself
        if value is None:
            return key

        # Format if string
        if isinstance(value, str) and kwargs:
            try:
                return value.format(**kwargs)
            except:
                return value

        return str(value)

    def set_language(self, lang_code: str) -> None:
        """
        Set the current language.

        Args:
            lang_code: Language code to switch to
        """
        if lang_code in self.available_languages:
            self.load_language(lang_code)
            # Store in session state for persistence
            if 'language' not in st.session_state:
                st.session_state.language = lang_code
            else:
                st.session_state.language = lang_code
        else:
            print(f"Warning: Language '{lang_code}' not available")

    def get_current_language(self) -> str:
        """Get current language code."""
        # Check session state first
        if hasattr(st, 'session_state') and 'language' in st.session_state:
            return st.session_state.language
        return self.current_language

    def get_available_languages(self) -> Dict[str, str]:
        """Get dictionary of available languages."""
        return self.available_languages


# Global translation manager instance
_translation_manager = None


def get_translation_manager() -> TranslationManager:
    """
    Get or create the global translation manager instance.

    Returns:
        TranslationManager instance
    """
    global _translation_manager
    if _translation_manager is None:
        _translation_manager = TranslationManager()
    return _translation_manager


def t(key: str, **kwargs) -> str:
    """
    Shorthand function for getting translations.

    Usage:
        t('pages.overview')
        t('metrics.accuracy')
        t('messages.welcome', name='John')

    Args:
        key: Translation key
        **kwargs: Optional formatting arguments

    Returns:
        Translated string
    """
    manager = get_translation_manager()
    return manager.get_translation(key, **kwargs)


def set_language(lang_code: str) -> None:
    """
    Set the current language.

    Args:
        lang_code: Language code (e.g., 'en', 'kk')
    """
    manager = get_translation_manager()
    manager.set_language(lang_code)


def get_current_language() -> str:
    """Get current language code."""
    manager = get_translation_manager()
    return manager.get_current_language()


def get_available_languages() -> Dict[str, str]:
    """Get available languages."""
    manager = get_translation_manager()
    return manager.get_available_languages()


def language_selector_sidebar() -> None:
    """
    Render language selector in Streamlit sidebar.
    This is a convenience function for easy integration.
    """
    manager = get_translation_manager()
    languages = manager.get_available_languages()

    if len(languages) <= 1:
        # Only one language available, no need for selector
        return

    current = manager.get_current_language()

    st.sidebar.markdown("---")
    st.sidebar.subheader("🌐 " + t('settings.language'))

    # Create language selector
    selected = st.sidebar.selectbox(
        label=t('settings.select_language'),
        options=list(languages.keys()),
        format_func=lambda x: languages[x],
        index=list(languages.keys()).index(current) if current in languages else 0,
        key='language_selector',
        label_visibility='collapsed'
    )

    # Update language if changed
    if selected != current:
        set_language(selected)
        st.rerun()
