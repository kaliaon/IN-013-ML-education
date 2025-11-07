# Translation Files (Locales)

This directory contains translation files for the dashboard in JSON format.

## 📁 File Structure

```
locales/
├── en.json          # English (default)
├── kk.json          # Kazakh (Қазақша)
└── README.md        # This file
```

## 🌍 Adding a New Language

To add a new language (e.g., Russian):

1. **Copy the English template:**
   ```bash
   cp en.json ru.json
   ```

2. **Update the language name:**
   ```json
   {
     "_language_name": "Русский",
     ...
   }
   ```

3. **Translate all values** (keep keys in English)

4. **Test:** Language will appear automatically in the dashboard

## 📝 Translation File Format

```json
{
  "_language_name": "Language Display Name",
  "_comment": "Optional comment",

  "section": {
    "key": "Translated text",
    "nested": {
      "key": "Deeply nested translation"
    }
  }
}
```

### Special Keys:
- `_language_name`: How the language appears in the selector
- `_comment`: Optional description (not used in UI)

## 🔑 Key Naming Conventions

Keys use dot notation for nesting:
- `app.title` → Title of the application
- `pages.overview` → Overview page name
- `metrics.accuracy` → Accuracy metric label

### Key Categories:
- **app.***: Application-level text
- **pages.***: Page titles and navigation
- **metrics.***: Measurement labels
- **common.***: Reusable UI elements
- **messages.***: User notifications
- **help.***: Help text and descriptions

## 💡 Translation Guidelines

### Do Translate:
- ✅ All user-facing text
- ✅ Button labels
- ✅ Error messages
- ✅ Help text
- ✅ Metric names

### Don't Translate:
- ❌ JSON keys (keep in English)
- ❌ Technical terms (optional: ML, ROC-AUC, etc.)
- ❌ File names
- ❌ Code references

### Best Practices:
1. **Keep formatting placeholders:** `{name}` stays as `{name}`
2. **Preserve special characters:** Icons like 🏠, 🎯 are universal
3. **Match tone:** Keep professional and clear
4. **Test in UI:** See how text fits in the interface
5. **Context matters:** Consider where the text appears

## 🔧 Usage in Code

The i18n system automatically loads translations:

```python
from visualization.i18n import t

# Simple translation
title = t('app.title')

# With formatting
message = t('messages.welcome', name='John')

# Nested keys
label = t('metrics.accuracy')
```

## 📊 Current Translation Status

| Language | Code | Status | Completeness |
|----------|------|--------|--------------|
| English | en | ✅ Complete | 100% |
| Kazakh | kk | 🚧 In Progress | ~80% |

## 🎯 Priority Translation Areas

When translating, focus on these high-visibility areas first:

1. **Navigation & Pages** (pages.*)
2. **Common UI Elements** (common.*)
3. **Metrics & Outcomes** (metrics.*, outcomes.*)
4. **System Messages** (messages.*)
5. **Page-specific content** (overview.*, predictions.*, etc.)

## 🔄 Updating Translations

When adding new features:

1. Add keys to `en.json` first (source of truth)
2. Update other language files
3. Use descriptive keys: `new_feature.button_label` not `btn1`
4. Document new keys in this README if needed

## 🧪 Testing Translations

1. **Start dashboard:** `python run_dashboard.py`
2. **Switch language:** Use sidebar selector
3. **Check all pages:** Verify text appears correctly
4. **Test formatting:** Ensure placeholders work
5. **Look for missing keys:** Untranslated keys show as `key.name`

## 📖 Translation Tools (Optional)

For large-scale translation, consider:
- **Google Translate API** (bulk translation)
- **DeepL** (higher quality)
- **Professional translator** (for dissertation)
- **Native speaker review** (quality assurance)

## 🤝 Contributing Translations

If adding or improving translations:

1. Edit the appropriate `.json` file
2. Keep the same structure as `en.json`
3. Test in the dashboard
4. Update this README if adding new language

## 📝 Notes

- **Encoding:** All files must be UTF-8
- **Format:** Valid JSON (use validator if unsure)
- **Line length:** Keep translations readable
- **Comments:** Use `_comment` keys for notes
- **Fallback:** English is used if translation missing

## 🎓 For Dissertation

The Kazakh translation (`kk.json`) is partially complete with key academic and interface terms. Priority was given to:
- Navigation elements
- Metrics and outcomes
- Common UI patterns
- System messages

Detailed content descriptions and help text may remain in English unless specifically required for defense.

---

**Last Updated:** 2025-01-06
**Maintained By:** Project Team
**Contact:** See main project README
