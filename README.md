<div align="center">

# 🔤 Autocomplete & Autocorrect Analytics

### 🤖 Exploring Intelligent Text Suggestion, Spelling Correction & NLP Techniques with Python

**OASIS INFOBYTE INTERNSHIP — LEVEL 2 | PROJECT 5**

[![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge\&logo=python)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge\&logo=jupyter)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge\&logo=pandas)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical_Computing-013243?style=for-the-badge\&logo=numpy)](https://numpy.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-85A0A6?style=for-the-badge)](https://www.nltk.org/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Machine_Learning-F7931E?style=for-the-badge\&logo=scikit-learn)](https://scikit-learn.org/)
[![Gensim](https://img.shields.io/badge/Gensim-Word2Vec-4B8BBE?style=for-the-badge)](https://radimrehurek.com/gensim/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge\&logo=streamlit)](https://streamlit.io/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0?style=for-the-badge)](https://seaborn.pydata.org/)

</div>

---

## 📌 Project Overview

**Autocomplete & Autocorrect Analytics** is an exploratory Python-based project developed during the **OASIS INFOBYTE Internship — Level 2, Project 5**.

The project investigates how intelligent text suggestion and spelling-correction functionality can be developed using different Python and Natural Language Processing techniques.

Rather than relying on a single algorithm, the notebook explores several approaches ranging from simple similarity-based correction to **N-gram concepts, Word2Vec contextual suggestions, machine-learning demonstrations, user feedback, benchmarking, visualization, and a Streamlit-based interface prototype**.

The project is implemented primarily as a large **Jupyter Notebook**, containing data analysis, preprocessing functions, algorithm experiments, demonstrations, and visualization.

---

## 🎯 Objectives

The major objectives of the project are:

* 🔤 Explore **autocomplete** functionality using prefix-based matching.
* ✏️ Implement basic **autocorrect** using string similarity.
* 🧹 Investigate basic **NLP preprocessing** techniques.
* 📊 Experiment with **N-gram-based autocomplete concepts**.
* 🧠 Explore **Word2Vec-based contextual word suggestions**.
* 💬 Simulate **user feedback collection**.
* 🌐 Develop a prototype **Streamlit user interface**.
* ⚡ Measure autocomplete execution time.
* 📈 Visualize suggestion-related results.
* 🤖 Experiment with machine-learning techniques such as **Multinomial Naive Bayes**.
* 🔍 Analyze the limitations of applying NLP techniques to a dataset that is primarily numerical.

---

# 🧠 Project Concept

Autocomplete and autocorrect systems are commonly used in:

* Search engines
* Mobile keyboards
* Web forms
* Code editors
* Messaging applications
* Search bars
* Digital assistants

The basic idea explored in this project is:

```text
User Input
     │
     ▼
Text / Prefix
     │
     ├───────────────┐
     ▼               ▼
Autocomplete      Autocorrect
     │               │
     ▼               ▼
Candidate Words   Similar Words
     │               │
     └───────┬───────┘
             ▼
       Suggested Output
             │
             ▼
       User Feedback
             │
             ▼
      Performance Analysis
```

---

# 🔬 Approaches Explored

The notebook contains multiple experimental approaches.

| Approach                  | Main Technique                | Purpose                                              |
| ------------------------- | ----------------------------- | ---------------------------------------------------- |
| 🔤 Basic Autocorrect      | `difflib.get_close_matches()` | Find similar words                                   |
| 🧹 NLP Preprocessing      | NLTK                          | Tokenization, lowercase conversion, stopword removal |
| 🔢 N-gram Autocomplete    | `CountVectorizer`             | Explore N-gram representation                        |
| 🧠 Contextual Suggestions | Word2Vec                      | Explore semantic similarity                          |
| 💬 Feedback Collection    | Python `input()`              | Simulate user feedback                               |
| 🌐 User Interface         | Streamlit                     | Prototype interactive application                    |
| 🤖 ML Demonstration       | Multinomial Naive Bayes       | Experiment with text classification                  |
| ⚡ Benchmarking            | Python `time`                 | Measure execution time                               |
| 📊 Visualization          | Matplotlib + Seaborn          | Analyze and visualize results                        |

---

# ✏️ 1. Basic Autocorrect

The first approach uses Python's built-in `difflib` functionality.

```python
from difflib import get_close_matches

def autocorrect(word, possibilities):
    return get_close_matches(
        word,
        possibilities,
        n=1,
        cutoff=0.8
    )
```

The function compares the input word with a list of candidate words and attempts to return the closest match.

Example:

```python
word_to_correct = "exampel"

corrected_word = autocorrect(
    word_to_correct,
    data['V1'].tolist()
)
```

### Example Output

```text
Suggested correction for 'exampel': []
```

The empty result is important because it demonstrates a limitation of the selected candidate vocabulary: the dataset column being used does not contain appropriate natural-language words for the test input.

---

# 🧹 2. NLP Preprocessing

The notebook also defines a basic NLP preprocessing pipeline using **NLTK**.

```python
def preprocess_text(text):
    words = word_tokenize(text.lower())

    words = [
        word
        for word in words
        if word.isalpha()
        and word not in stopwords.words('english')
    ]

    return ' '.join(words)
```

The preprocessing function demonstrates:

* Lowercase conversion
* Word tokenization
* Alphabetic-token filtering
* Stopword removal
* Reconstruction of cleaned text

The notebook also includes an example application to a potential text column.

> **Note:** The demonstrated preprocessing function is defined in the notebook, while its application to the original dataset is not the primary processing path because the dataset contains numerical anonymized features rather than conventional text.

---

# 🔢 3. N-gram Autocomplete Experiment

The notebook explores N-gram-based autocomplete using:

```python
from sklearn.feature_extraction.text import CountVectorizer
```

A dedicated `NGramAutocomplete` class is created.

```python
class NGramAutocomplete:

    def __init__(self, data, n=2):
        self.data = data
        self.n = n

        if not all(isinstance(text, str) for text in self.data):
            raise ValueError(
                "All items in the data must be strings."
            )

        self.vectorizer = CountVectorizer(
            ngram_range=(n, n),
            lowercase=True
        )

        self.vectorizer.fit(data)

    def predict(self, prefix):
        matches = [
            text
            for text in self.data
            if text.startswith(prefix)
        ]

        return matches
```

A sample text collection was created:

```text
example
examine
exam
exercise
extra
```

The data was first cleaned by removing:

* Missing values
* Empty strings
* Whitespace-only entries

### Experimental Result

The experiment produced:

```text
Error: empty vocabulary; perhaps the documents only contain stop words
```

This result is documented as an **experimental limitation** rather than being presented as a successful trained N-gram model.

---

# 🧠 4. Contextual Autocorrection with Word2Vec

The project further explores contextual word representation using **Gensim Word2Vec**.

The basic workflow is:

```text
Text Data
   │
   ▼
Tokenization
   │
   ▼
Word2Vec Training
   │
   ▼
Word Embeddings
   │
   ▼
Similar Word Search
   │
   ▼
Contextual Suggestions
```

The notebook creates a Word2Vec model:

```python
model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)
```

A contextual suggestion function is then defined:

```python
def contextual_autocorrect(word):

    if word in model.wv:
        similar_words = model.wv.most_similar(
            word,
            topn=5
        )

        return [
            word
            for word, _ in similar_words
        ]

    return []
```

### Experimental Result

For:

```text
exampel
```

the notebook returned:

```text
Contextual suggestions for 'exampel': []
```

This occurs because the input word is not present in the learned vocabulary.

---

# ⚠️ 5. Dataset Consideration

The project uses the **Credit Card Fraud Detection** dataset as its underlying CSV dataset.

The dataset contains:

```text
Time
V1 – V28
Amount
Class
```

The anonymized `V1`–`V28` features are numerical rather than conventional natural-language text.

For example:

```text
V1 = -1.359807
V2 = -0.072781
V3 = 2.536347
...
```

This creates an important experimental limitation for an NLP-oriented project.

### Why this matters

Autocomplete and autocorrect normally require a natural-language corpus such as:

```text
example
examine
exam
exercise
extra
```

whereas the selected dataset contains values such as:

```text
-1.359807
0.266151
-1.340163
```

Therefore, several notebook sections use **sample text data or demonstration datasets** to explore the intended autocomplete/autocorrect techniques.

This distinction is deliberately documented because the notebook is an **exploratory implementation and learning project**, rather than a production NLP system trained on a dedicated language corpus.

---

# 💬 6. User Feedback Simulation

The project also explores how user feedback could be incorporated into an intelligent suggestion system.

```python
def gather_feedback(predictions, user_input):

    feedback = input(
        f"Did the suggestions {predictions} help? (yes/no): "
    )

    return feedback.lower() == "yes"
```

Example:

```text
Feedback received: True
```

The implementation demonstrates the concept of collecting user feedback.

In a production application, this could instead be connected to:

* A web interface
* Database storage
* User interaction logs
* Model evaluation pipelines
* Recommendation improvement mechanisms

---

# 🌐 7. Streamlit User Interface

The notebook contains a prototype **Streamlit interface**.

The intended interface provides separate inputs for:

### 🔤 Autocomplete

```python
user_input = st.text_input(
    "Enter text for autocomplete:"
)
```

### ✏️ Autocorrect

```python
user_word = st.text_input(
    "Enter a word for autocorrect:"
)
```

The interface is designed to display:

```text
User Input
    │
    ├──► Autocomplete Suggestions
    │
    └──► Autocorrect Suggestions
```

This moves the project beyond a purely notebook-based experiment toward an interactive application concept.

---

# 🤖 8. Machine Learning Experiment — Multinomial Naive Bayes

The notebook also includes an experimental machine-learning implementation using:

```python
from sklearn.naive_bayes import MultinomialNB
```

A small demonstration dataset was created:

```python
data = [
    "apple",
    "banana",
    "grape",
    "orange",
    "strawberry"
]
```

The corresponding demonstration target was:

```python
y = ["fruit"] * len(data)
```

Text features were generated using:

```python
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data)
```

A Multinomial Naive Bayes model was then trained:

```python
autocomplete_model = MultinomialNB()
autocomplete_model.fit(X, y)
```

### Important distinction

This is a **demonstration experiment**, not a validated autocomplete model.

The target labels are intentionally simple:

```text
apple       → fruit
banana      → fruit
grape       → fruit
orange      → fruit
strawberry  → fruit
```

The experiment demonstrates the mechanics of:

```text
Text
 ↓
CountVectorizer
 ↓
Numerical Features
 ↓
Multinomial Naive Bayes
 ↓
Prediction
```

---

# ⚡ 9. Performance Benchmarking

The project includes execution-time benchmarking using Python's `time` module.

```python
def benchmark_function(func, *args):

    start_time = time.time()

    result = func(*args)

    end_time = time.time()

    execution_time = end_time - start_time

    return result, execution_time
```

Example:

```text
Autocomplete Result: ['apple']
Autocomplete Time: 0.0000 seconds
```

This provides a basic way to compare the execution time of suggestion functions.

---

# 📊 10. Data Visualization

The notebook also contains visualization experiments using:

* Matplotlib
* Seaborn
* Pandas

One example visualizes the distribution of the `Amount` column:

```python
sns.histplot(
    data['Amount'],
    bins=30,
    kde=True
)
```

The notebook also attempts to visualize autocomplete suggestions using a Seaborn count plot.

If no suggestions are available, the notebook explicitly reports:

```text
No autocomplete suggestions found for the prefix 'exam'.
```

This makes the visualization dependent on the output generated by the preceding autocomplete experiment.

---

# 🔄 Overall Experimental Workflow

```text
                 ┌─────────────────────┐
                 │     CSV Dataset     │
                 │ Credit Card Dataset │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Data Inspection &   │
                 │ Preprocessing       │
                 └──────────┬──────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │      Text Experiments      │
              └─────────────┬───────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
   Difflib           N-gram Model         Word2Vec
 Autocorrect        Autocomplete       Contextual NLP
         │                  │                  │
         └──────────────────┼──────────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Suggestion Output   │
                 └──────────┬──────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
          Feedback       Benchmark      Visualization
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                   ┌────────────────┐
                   │ Streamlit UI   │
                   └────────────────┘
```

---

# 🛠️ Technologies & Libraries

| Technology              | Purpose                               |
| ----------------------- | ------------------------------------- |
| 🐍 **Python**           | Core programming language             |
| 📓 **Jupyter Notebook** | Development and experimentation       |
| 🐼 **Pandas**           | Data loading and manipulation         |
| 🔢 **NumPy**            | Numerical computing                   |
| 🧠 **NLTK**             | Natural Language Processing           |
| 🔍 **difflib**          | Similarity-based autocorrection       |
| 📊 **Scikit-learn**     | Feature extraction and ML experiments |
| 🤖 **MultinomialNB**    | Machine-learning demonstration        |
| 🧠 **Gensim**           | Word2Vec experimentation              |
| 🌐 **Streamlit**        | Interactive UI prototype              |
| 📈 **Matplotlib**       | Data visualization                    |
| 📊 **Seaborn**          | Statistical visualization             |

---

# 📁 Repository Structure

```text
OIBSIP-Intern-Saswati-Autocomplete-Autocorrect-Analytics
│
├── AUTOCORRECT.ipynb
│
├── AUTOCORRECT_DATSET LINK.txt
│
└── README.md
```

### Main Notebook

`AUTOCORRECT.ipynb`

Contains the complete experimentation workflow, including:

* Dataset inspection
* NLP preprocessing
* Autocorrect
* Autocomplete
* N-gram experiments
* Word2Vec
* Feedback simulation
* Streamlit prototype
* Naive Bayes experiment
* Benchmarking
* Visualization

---

# 🚀 Getting Started

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/SASWATIMATHAN/OIBSIP-Intern-Saswati-Autocomplete-Autocorrect-Analytics-PROJECT-5-LEVEL-2.git
```

```bash
cd OIBSIP-Intern-Saswati-Autocomplete-Autocorrect-Analytics-PROJECT-5-LEVEL-2
```

---

## 2️⃣ Install Required Libraries

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn gensim streamlit
```

For NLTK resources:

```python
import nltk

nltk.download('punkt')
nltk.download('stopwords')
```

---

## 3️⃣ Launch the Notebook

```bash
jupyter notebook
```

Open:

```text
AUTOCORRECT.ipynb
```

and execute the cells sequentially.

---

# 🌐 Running the Streamlit Prototype

If the Streamlit section has been separated into a Python application file, it can be launched with:

```bash
streamlit run app.py
```

The prototype is intended to provide an interface for:

```text
Input
  ↓
Autocomplete / Autocorrect
  ↓
Suggestions
  ↓
User Feedback
```

---

# 📈 Key Learning Outcomes

Through this project, the following concepts were explored:

### 🐍 Python

* Functions
* Lists
* String processing
* Exception handling
* Input handling
* Execution-time measurement

### 🧠 Natural Language Processing

* Tokenization
* Stopword removal
* Lowercasing
* Text preprocessing
* N-grams
* Word embeddings
* Contextual similarity

### 🤖 Machine Learning

* CountVectorizer
* Feature representation
* Multinomial Naive Bayes
* Word2Vec
* Model prediction

### 📊 Data Analysis

* CSV loading
* Missing-value inspection
* Data cleaning
* Data type inspection
* Statistical visualization

### 🌐 Application Development

* Streamlit interface concepts
* User input handling
* Suggestion display
* Feedback collection

### ⚡ Performance Analysis

* Function benchmarking
* Execution-time measurement

---

# ⚠️ Experimental Limitations

This project should be understood as an **exploratory internship project**, not as a production-ready NLP autocomplete engine.

Several experiments reveal limitations that are useful from a learning perspective.

### 1. Dataset mismatch

The primary Credit Card Fraud Detection dataset contains anonymized numerical features rather than a natural-language corpus.

### 2. Autocorrect vocabulary

The `difflib` implementation depends heavily on the candidate vocabulary. Using numerical dataset values as candidate words does not provide meaningful natural-language correction.

### 3. N-gram experiment

The N-gram implementation encountered an empty-vocabulary error during the demonstrated experiment.

### 4. Word2Vec experiment

The Word2Vec implementation did not produce suggestions for `"exampel"` because the queried word was unavailable in the learned vocabulary.

### 5. Demonstration ML dataset

The Multinomial Naive Bayes section uses a very small artificial fruit dataset and therefore should be interpreted as an algorithm demonstration rather than a production-trained autocomplete model.

### 6. Feedback mechanism

Feedback is simulated using Python's `input()` function and is not persistently stored.

---

# 🔮 Future Improvements

A production-oriented version could improve the project substantially by replacing the experimental dataset and extending the NLP pipeline.

### 📚 Dedicated Text Corpus

Use a genuine language dataset containing:

```text
sentences
paragraphs
search queries
word frequencies
conversation text
```

### 🧠 Improved Autocomplete

Possible approaches:

* Frequency-based autocomplete
* Character N-grams
* Word N-grams
* Trie-based prefix search
* Language models
* Transformer-based models

### ✏️ Improved Autocorrect

Possible enhancements:

* Edit-distance algorithms
* Frequency-aware correction
* Phonetic similarity
* Context-aware correction
* Candidate ranking

### 💬 Persistent Feedback

Store:

```text
User Input
Suggested Words
Selected Suggestion
Accepted / Rejected
Timestamp
```

and use this information to improve future suggestions.

### 🌐 Complete Web Application

A future implementation could combine:

```text
Python Backend
      ↓
NLP / ML Engine
      ↓
REST API
      ↓
Streamlit / Web Frontend
      ↓
User Feedback
      ↓
Database
```

---

# 🏆 Project Significance

This project demonstrates an important progression from **basic string matching to experimental NLP and machine-learning approaches**.

The notebook explores the complete conceptual pipeline:

```text
Data
 ↓
Preprocessing
 ↓
Candidate Generation
 ↓
Autocomplete / Autocorrect
 ↓
Contextual Modeling
 ↓
User Interaction
 ↓
Feedback
 ↓
Benchmarking
 ↓
Visualization
```

Although some experiments did not produce useful suggestions because of dataset and vocabulary limitations, those results provide valuable insight into the importance of **choosing an appropriate corpus, representation, vocabulary, and evaluation strategy** when developing NLP applications.

---

# 🎓 Internship Context

**Program:** OASIS INFOBYTE Internship

**Level:** Level 2

**Project:** Project 5 — Autocomplete / Autocorrect Analytics

**Primary Language:** Python

**Development Environment:** Jupyter Notebook

**Application Component:** Streamlit Prototype

---

# 👩‍💻 Author

### **Saswati Anupama Mathan**

**M.Tech — Electronics & Communication Engineering**

GitHub: `SASWATIMATHAN`

---

# 🙏 Acknowledgements

Special thanks to **OASIS INFOBYTE** for providing the internship opportunity and project framework through which this Python, NLP, machine-learning, and application-development work was explored.

---

<div align="center">

### ⭐ If you find this project interesting, consider giving the repository a star!

**Built with Python • NLP • Machine Learning • Data Analysis • Streamlit**

</div>
