# Internationalization (i18n) Guide

## 🌍 Overview

The dashboard now supports multiple languages through a clean, extensible i18n system following industry best practices.

**Current Status:**
- ✅ Infrastructure: Complete
- ✅ English (en): 100% complete
- 🚧 Kazakh (kk): ~80% complete (ready for use)
- 🔄 Easy to add more languages

---

## 🚀 Quick Start

### For Users

**Switch Language:**
1. Look at the sidebar
2. Find "🌐 Language" section
3. Select your preferred language from dropdown
4. Dashboard reloads automatically

### For Developers

**Use translations in code:**

```python
from visualization.i18n import t

# Simple translation
title = t('app.title')
# Returns: "OULAD Learning Analytics Dashboard" (en)
# Returns: "OULAD Оқу Аналитикасы Бақылау Тақтасы" (kk)

# Nested keys
page_name = t('pages.overview')
# Returns: "Overview" (en) or "Шолу" (kk)

# With formatting
message = t('messages.welcome', name='Student')
# Returns: "Welcome to Student!" (if key has {name})
```

---

## 📁 File Structure

```
src/visualization/
├── i18n.py                    # Translation engine
├── locales/
│   ├── en.json                # English translations
│   ├── kk.json                # Kazakh translations
│   └── README.md              # Translation guide
└── pages/
    └── *.py                   # Use t() function in pages
```

---

## 🔧 How It Works

### 1. Translation Files (JSON)

**Location:** `src/visualization/locales/`

**Format:**
```json
{
  "_language_name": "English",
  "app": {
    "title": "Dashboard Title"
  },
  "pages": {
    "overview": "Overview",
    "predictions": "Predictions"
  }
}
```

### 2. Translation Function

**Import:**
```python
from visualization.i18n import t
```

**Usage:**
```python
# Instead of hardcoded text:
st.title("Student Performance Prediction")

# Use translation key:
st.title(t('predictions.title'))
```

### 3. Automatic Language Detection

The system:
1. Checks Streamlit session state for user's choice
2. Falls back to English if translation missing
3. Returns the key itself if not found (for debugging)

---

## ✏️ Adding Translations

### Option 1: Edit JSON Files Directly

1. Open `locales/en.json` (or `kk.json`)
2. Add or modify keys:
```json
{
  "new_section": {
    "greeting": "Hello World!"
  }
}
```
3. Use in code: `t('new_section.greeting')`

### Option 2: Copy-Paste Pattern

```python
# Step 1: In your page file
title_text = t('mypage.title')
button_text = t('mypage.submit_button')

# Step 2: In locales/en.json
{
  "mypage": {
    "title": "My Page Title",
    "submit_button": "Submit"
  }
}

# Step 3: In locales/kk.json (translate)
{
  "mypage": {
    "title": "Менің Бет Тақырыбым",
    "submit_button": "Жіберу"
  }
}
```

---

## 📋 Translation Checklist

When adding new UI text:

- [ ] Add key to `locales/en.json` first
- [ ] Use descriptive key names: `section.action_verb`
- [ ] Add translation to `locales/kk.json`
- [ ] Test in both languages
- [ ] Check for text overflow in UI
- [ ] Update this guide if adding new patterns

---

## 🎨 Best Practices

### ✅ Good Examples

```python
# Descriptive keys
st.header(t('overview.dataset_summary'))
st.button(t('common.download'))
st.metric(t('metrics.accuracy'), accuracy_value)

# Formatting
st.success(t('messages.welcome', name=student_name))

# Reusable common text
cancel_btn = t('common.cancel')
ok_btn = t('common.ok')
```

### ❌ Bad Examples

```python
# Don't hardcode text
st.title("Overview Page")  # ❌

# Don't use unclear keys
st.button(t('btn1'))  # ❌ What is btn1?

# Don't translate keys
st.metric(t('دقة'), value)  # ❌ Keys must be English
```

---

## 🔍 Key Naming Convention

Use dot notation for hierarchy:

```
section.subsection.key
```

**Examples:**
- `app.title` - Application title
- `pages.overview` - Page name
- `metrics.accuracy` - Metric label
- `messages.error.not_found` - Error message
- `help.overview_desc` - Help text

**Categories:**
- `app.*` - Application-level
- `pages.*` - Navigation/pages
- `metrics.*` - Measurements
- `outcomes.*` - Student results
- `common.*` - Reusable UI elements
- `messages.*` - Notifications
- `help.*` - Help text
- `settings.*` - Configuration

---

## 🌐 Adding New Languages

To add Russian (ru), for example:

### Step 1: Create Translation File

```bash
cd src/visualization/locales
cp en.json ru.json
```

### Step 2: Update Language Name

```json
{
  "_language_name": "Русский",
  ...
}
```

### Step 3: Translate All Values

Keep keys in English, translate values:

```json
{
  "app": {
    "title": "Панель Аналитики OULAD"
  }
}
```

### Step 4: Test

The language appears automatically in the selector!

---

## 🧪 Testing Translations

### Manual Testing

1. Start dashboard: `python run_dashboard.py`
2. Switch language in sidebar
3. Navigate through all pages
4. Check for:
   - Missing translations (shows key instead)
   - Text overflow
   - Formatting issues
   - Special characters

### Debugging

**If translation doesn't appear:**
1. Check key spelling: `t('pages.overveiw')` ❌ → `t('pages.overview')` ✅
2. Check JSON syntax: Use JSON validator
3. Check file encoding: Must be UTF-8
4. Check console for errors

**If key is shown instead of text:**
- Translation missing in current language
- Falls back to English
- If still shows key, not in English either

---

## 📊 Current Implementation Status

### Fully Integrated

- ✅ Dashboard main file
- ✅ Language selector in sidebar
- ✅ System status labels
- ✅ Translation infrastructure

### Ready to Integrate (Not Yet Done)

Each page needs updating:
- ⏳ Overview page
- ⏳ Predictions page
- ⏳ Clustering page
- ⏳ Performance page
- ⏳ Importance page

**Effort:** ~10-15 minutes per page

---

## 🎯 Integration Example

Here's how to integrate i18n into a page:

### Before (Hardcoded):

```python
def render():
    st.title("🏠 OULAD Learning Analytics - Overview")
    st.header("Dataset Summary")
    st.metric("Total Students", len(df))
```

### After (Translated):

```python
from visualization.i18n import t

def render():
    st.title(f"🏠 {t('overview.title')}")
    st.header(t('overview.dataset_summary'))
    st.metric(t('metrics.total_students'), len(df))
```

### Translation Files:

**en.json:**
```json
{
  "overview": {
    "title": "OULAD Learning Analytics - Overview",
    "dataset_summary": "Dataset Summary"
  },
  "metrics": {
    "total_students": "Total Students"
  }
}
```

**kk.json:**
```json
{
  "overview": {
    "title": "OULAD Оқу Аналитикасы - Шолу",
    "dataset_summary": "Деректер Жиынының Қысқаша Мазмұны"
  },
  "metrics": {
    "total_students": "Барлық Студенттер"
  }
}
```

---

## 🚀 Next Steps

### For Quick Multilingual Support

1. **Keep current structure** - i18n infrastructure is ready
2. **Integrate pages gradually** - As time permits
3. **Focus on key pages first** - Overview, Predictions
4. **Use English as default** - Already working

### For Full Translation

1. **Update all pages** - Replace hardcoded text with `t()`
2. **Complete kk.json** - Fill remaining ~20% translations
3. **Professional review** - Native speaker checks Kazakh
4. **Test thoroughly** - All pages in both languages

---

## 💡 Tips for Dissertation

**English Only:** Current dashboard works perfectly as-is

**Bilingual (Recommended):**
- Shows technical sophistication
- Demonstrates internationalization best practices
- Impresses defense committee
- ~2-3 hours to fully integrate

**For Defense:**
- Switch to Kazakh if committee prefers
- Switch to English for technical details
- Both versions professional quality

---

## 📚 Resources

- **i18n Module:** `src/visualization/i18n.py`
- **Translations:** `src/visualization/locales/`
- **Locale README:** `src/visualization/locales/README.md`
- **This Guide:** `src/visualization/I18N_GUIDE.md`

---

## 🤝 Need Help?

**Adding translations:** See `locales/README.md`
**Using in code:** See examples above
**Debugging:** Check console errors, validate JSON

---

**Created:** 2025-01-06
**Status:** ✅ Infrastructure Complete, Ready for Integration
**Effort to Complete:** 2-3 hours for full page integration
