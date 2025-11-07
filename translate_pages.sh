#!/bin/bash
# Script to translate all page modules to use i18n

cd "$(dirname "$0")/src/visualization/page_modules"

echo "🔄 Translating page modules..."

# First, add the import statement to each page if not already present
for page in predictions.py clustering.py performance.py importance.py; do
    if ! grep -q "from visualization.i18n import t" "$page"; then
        # Find the last import line and add the i18n import after it
        sed -i '/^from visualization/a from visualization.i18n import t' "$page"
        echo "✅ Added i18n import to $page"
    fi
done

echo "✅ Translation imports added to all pages"
echo "📝 Note: Manual text replacement needed for complex expressions"
echo "   Run the dashboard and check each page for remaining untranslated text"
