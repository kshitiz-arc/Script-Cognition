# 🌐 Web App User Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install streamlit pandas
```

Or upgrade existing installation:
```bash
pip install -r requirements.txt
```

### 2. Launch the Web App

**Option A (Recommended):**
```bash
python run_app.py
```

**Option B (Direct Streamlit):**
```bash
streamlit run app.py
```

### 3. Access the App

The app will automatically open in your browser at:
```
http://localhost:8501
```

If not, manually open the URL in your browser.

---

## Web App Features

### 🏠 Home Page

**Overview of the project**
- Welcome message
- Quick start instructions
- Model status (loaded/not loaded)
- Dataset status
- Target emotion info

**What to do:**
1. Check if model is loaded (green = ready)
2. Click "Test Handwriting" to make predictions

### 📊 Dataset Explorer

**Explore the EMOTHAW dataset**

**Features:**
- View dataset statistics (users, samples, tasks)
- See class distribution charts
- Browse sample handwriting trajectories
- Compare pen status vs pressure visualization

**How to use:**
1. Select User ID (1, 2, 3, ...)
2. Select Task ID (1-7)
3. Click "Display Trajectory" button
4. View 2 plots:
   - Left: Pen DOWN/UP status
   - Right: Pressure heatmap (color intensity = pressure)

### 🧪 Test Handwriting (Main Feature)

**Analyze handwriting emotion**

#### Input Options

**Option 1: Upload .svc File**
- Click "Browse files"
- Select a .svc handwriting file
- File is displayed with datapoint count

**Option 2: Select from Dataset**
- Enter User ID (1+)
- Enter Task ID (1-7)
- Dataset sample is loaded
- Actual label shown for comparison

#### Processing Buttons

**Three buttons available:**

##### 1. 🎯 Analyze Handwriting
**What it does:**
- Loads trained CNN model
- Processes handwriting
- Outputs emotion prediction

**Results shown:**
- Big emotion box (GREEN = Low, RED = High)
- Confidence score (0-100%)
- Data points analyzed
- Probability distribution table
- Bar chart showing class probabilities

**Download option:**
- Download results as CSV

##### 2. 📈 Visualize Trajectory
**What it does:**
- Renders pen trajectory visualization
- Shows 2 plots

**Left plot (Pen Status):**
- Navy dots = pen DOWN (writing)
- Red dots = pen UP (not writing)

**Right plot (Pressure Heatmap):**
- Same trajectory
- Color intensity = pen pressure
- Bright = high pressure, Dark = low pressure
- Colorbar shows pressure scale

##### 3. 🔍 Extract Features
**What it does:**
- Extracts 29+ handwriting features
- Organized in 4 tabs

**Tabs:**
- **Velocity**: Writing speed stats
- **Pressure**: Force variation
- **Trajectory**: Shape and size metrics
- **Other**: Pen lifts, pauses, efficiency

**Each feature shown as metric**

**Download option:**
- Export all features as CSV file

### ℹ️ Information Page

**Learn about the project**

**Expandable sections:**
1. **About This Project**
   - Dataset info
   - Model architecture
   - How it works

2. **How to Use**
   - Step-by-step guide
   - What is .svc format
   - Example file format

3. **Understanding Results**
   - Emotion classifications
   - Confidence score interpretation
   - Probability distribution

4. **Extracted Features (29 total)**
   - Feature categories
   - What each means
   - Emotional correlates

5. **FAQ**
   - Common questions
   - Troubleshooting

6. **Project Links**
   - File references
   - CLI commands

---

## Workflow Examples

### 🎯 Workflow 1: Quick Test (2 minutes)

1. **Home** → Check model is loaded
2. **Test Handwriting**
   - Select from Dataset (User 1, Task 1)
   - Click "Analyze Handwriting"
   - View prediction result
3. **Done!**

### 📊 Workflow 2: Explore Dataset (5 minutes)

1. **Dataset Explorer**
   - View statistics and class distribution
   - Select different users and tasks
   - Click "Display Trajectory"
   - Compare trajectories
2. **Learn patterns** in handwriting

### 🧪 Workflow 3: Upload & Analyze (3 minutes)

1. **Test Handwriting**
   - Click file uploader
   - Select your .svc file
   - Click "Analyze Handwriting"
   - See emotion prediction
2. **Optional:** Click "Extract Features" to see details

### 🔍 Workflow 4: Deep Analysis (10 minutes)

1. **Test Handwriting**
   - Upload .svc file or select from dataset
2. **Click all 3 buttons:**
   - Analyze → See prediction
   - Visualize → See trajectory
   - Extract → See raw features
3. **Download results** (CSV files)
4. **Analyze offline** with spreadsheet or python

---

## Understanding the Results

### Emotion Prediction

**Two possible outcomes:**
- **LOW** (Green box) = Normal or Mild emotion severity
- **HIGH** (Red box) = Moderate to Extremely Severe emotion

### Confidence Score

**Interpretation:**
- **90-100%**: Very confident prediction (trust it)
- **80-89%**: Confident (likely correct)
- **70-79%**: Moderate confidence (reasonable)
- **60-69%**: Low confidence (could be either class)
- **50-59%**: Uncertain (essentially guessing)

### Probability Distribution

**What it shows:**
- Probability for EACH class
- Should sum to 100%
- Larger gap = more confident

**Example:**
```
Low:  0.8412 (84.12%) ████████████████████████████
High: 0.1588 (15.88%) ████
```
Interpretation: Model is 84% sure it's LOW, 16% sure it's HIGH

### Extracted Features (29 total)

**Organized in 4 categories:**

1. **Velocity Features** (4)
   - How fast the pen moves
   - Faster = might indicate different emotions

2. **Pressure Features** (4)
   - How hard the person presses
   - Pressure variation = emotional state

3. **Trajectory Features** (5+)
   - Size, shape, coverage of writing
   - Larger writing = different emotion?

4. **Pen Behavior** (3)
   - How many times pen lifts
   - How long pauses are
   - Hesitation patterns

**All features are numerical** and can be used with ML models.

---

## Troubleshooting

### ❌ "No trained model found!"

**Solution:**
1. Open terminal in project directory
2. Run: `python main.py train --model cnn`
3. Wait for training to complete (5-10 minutes)
4. Refresh web app page

### ❌ File upload fails

**Possible causes:**
- File is not .svc format
- File is corrupted
- Wrong file extension

**Solution:**
- Verify file format is plain text
- Check file is readable with notepad
- Try example from dataset first

### ❌ App crashes or shows error

**Solution:**
1. Stop app (Ctrl+C)
2. Check Python version: `python --version` (need 3.10+)
3. Reinstall requirements: `pip install -r requirements.txt`
4. Restart: `python run_app.py`

### ⚠️ Slow results or predictions take long time

**Reason:**
- First run caches images
- CPU is slower than GPU

**Solution:**
- Be patient for first analysis
- Subsequent analyses are faster
- Use GPU if available

### ❌ Dataset not loading in Explorer

**Reason:**
- Dataset path incorrect
- DASS_scores.xls missing

**Solution:**
1. Go to Home page
2. Check "Dataset Found" status
3. Run: `python main.py info`
4. Check paths match

---

## Tips & Tricks

### ✨ Best Practices

1. **Start with Home page** to check status
2. **Explore Dataset first** to understand data
3. **Use simple examples** before complex ones
4. **Download results** for offline analysis
5. **Read Information page** for help

### 🎯 Key Buttons

| Button | When to Use | Time |
|--------|-------------|------|
| 🎯 Analyze | Get emotion prediction | <5s |
| 📈 Visualize | See pen movement | <5s |
| 🔍 Extract | Get 29 features | 1-2s |

### 📥 Download Options

All pages offer CSV export:
- Prediction results
- Feature values
- Statistics

**Use for:**
- Spreadsheet analysis
- Further research
- Model training (features)

### 🔄 Batch Processing

To test multiple files:
1. Analyze file 1, download results
2. Repeat for file 2, 3, etc.
3. Combine CSVs for analysis

Or use **CLI mode** for batch:
```bash
for file in *.svc; do
    python main.py predict --svc_path "$file"
done
```

---

## Advanced Usage

### Changing Emotions

Edit `config.py`:
```python
TARGET_EMOTION = "stress"  # anxiety, depression, or stress
```

Restart app:
```bash
python run_app.py
```

### Different Models

Train and switch models:
```bash
# Train ResNet
python main.py train --model resnet

# Restart app (it auto-loads best model)
python run_app.py
```

### Customize Appearance

Edit styles in `app.py`:
```python
st.markdown("""
    <style>
    .emotion-high { background-color: #YOUR_COLOR }
    ...
    </style>
""", unsafe_allow_html=True)
```

---

## Keyboard Shortcuts

| Action | Keyboard |
|--------|----------|
| Open sidebar | Ctrl+Shift+Backspace |
| Stop app | Ctrl+C (in terminal) |
| Refresh page | F5 or Ctrl+R |
| Toggle sidebar | Arrow button (top-left) |

---

## 📞 Getting Help

1. **Check Information page** (in web app)
2. **Read README.md** (project root)
3. **Run `python main.py info`** (diagnose problems)
4. **Check QUICK_START.md** (command reference)

---

## Next Steps

**After using web app:**
1. ✅ Understand emotions from handwriting
2. ✅ Explore dataset patterns
3. ✅ Test on your own data
4. ✅ Extract features for research
5. ✅ Consider retraining with more data

**To share results:**
- Download CSV files
- Share visualizations
- Document findings

---

**Happy analyzing! 🎉**
