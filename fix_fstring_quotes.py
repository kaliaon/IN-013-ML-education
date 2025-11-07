#!/usr/bin/env python3
"""
Fix escaped quotes in f-strings across all page modules.
"""

from pathlib import Path
import re

def fix_escaped_quotes(file_path: Path):
    """Replace t(\\'key\\') with t('key') throughout file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Replace all instances of t(\\'...\') with t('...')
    content = content.replace(r"t(\'", "t('")
    content = content.replace(r"\')", "')")

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Count replacements
        original_count = original_content.count(r"t(\'")
        print(f"✅ {file_path.name}: Fixed {original_count} escaped quotes")
        return True
    else:
        print(f"ℹ️  {file_path.name}: No changes needed")
        return False

def main():
    """Fix all page modules."""
    pages_dir = Path(__file__).parent / "src/visualization/page_modules"

    files_to_fix = [
        "predictions.py",
        "clustering.py",
        "performance.py",
        "importance.py"
    ]

    fixed_count = 0
    for filename in files_to_fix:
        file_path = pages_dir / filename
        if file_path.exists():
            if fix_escaped_quotes(file_path):
                fixed_count += 1
        else:
            print(f"⚠️  {filename}: File not found")

    print(f"\n🎉 Fixed {fixed_count} files!")
    print("✅ All f-string syntax errors should now be resolved.")

if __name__ == "__main__":
    main()
