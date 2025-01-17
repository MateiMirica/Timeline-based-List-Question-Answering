# Timeline-based-List-Question-Answering

## 1. Introduction
This repository contains code and resources for **Timeline-based List-Question Answering (TLQA)**. TLQA focuses on questions that 
* (1) require **list-based answers** and
* (2) incorporate **temporal constraints** or time periods. For example:

\
**Question**: "List all sports teams Robert Lewandowski played for from 2010 to 2020."

**Answer**:

* Lech Poznań (2010)
* Borussia Dortmund (2010-2014)
* FC Bayern Munich (2014-2020)

\
Our project explores the abilities of the **Flan-T5 model** to:

* Provide complete lists of answers.
* Correctly identify and align answers with time periods.


## 2. Research Questions
Our work investigates three main research questions:

* **RQ1**: How do fine-tuned generative models and few-shot prompting of generative models in a closed-book QA setting perform on TLQA?
* **RQ2**: How do fine-tuned generative models and few-shot prompting of generative models with retrieved top-k evidence perform on TLQA?
* **RQ3**: Does special handling of temporal markers (e.g., explicit time intervals in retrieval or generation) improve performance for TLQA?

## 3. Experiments and usage
Below we outline the main experiments described in our research. Detailed instructions for each experiment can be found in the scripts under `code/`.

### 3.1 Closed-book TLQA

#### Few-Shot Prompting

* Use the generative models FlanT5-large, FlanT5-XL.
* Implement a KNN-based sample selection for demonstrations. We use an embedding model (from sentence-transformers) to find k nearest neighbors in the training set that are similar to the test question.
* Vary k from {0, 3, 5, 7, 10} to measure performance changes.

#### Fine-Tuning

* Fine-tune models FlanT5-base andFlanT5-large.
* Run inference on the test set.
* Combine fine-tuning with few-shot prompting.

### 3.2 Retrieval-Augmented TLQA
* Retrieve top-k relevant contexts from a Wikipedia infobox collection.
* Provide the retrieved text as input to the generative model.
* Vary k from {1, 3, 5, 7, 10} to see how retrieval depth affects performance.
* Combine fine-tuning and few-shot prompting with rag.

### 3.3 Temporal Relevance in Retrieval
* Incorporate temporal relevance when searching for context in addition to simple cosine similarity.

## 4. Initial (one-time) Setup

Follow these steps to prepare your environment for the project:

### 4.1 Set up WSL (for Windows only)

If you don't already have WSL, run
```bash
wsl --install
```

Otherwise:
```bash
wsl
```

### 4.2 Navigate to the Project Directory

Change to the project's root directory:

```bash
cd <.../Timeline-based-List-Question-Answering>
```

### 4.3 Set Up a Virtual Environment

Create an isolated environment for your project:

1. **Create a virtual environment:**

```bash
python3 -m venv venv
```

2. **Activate the virtual environment:**
```bash
source venv/bin/activate
```

### 4.4 Install Required Packages

Install the necessary packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 5. Running after the initial setup
```bash
wsl
```
```bash
cd <.../Timeline-based-List-Question-Answering>
```
```bash
source venv/bin/activate
```
```bash
jupyter notebook
```

In the WSL window, you will see a link similar to  http://localhost:8889/tree?token=string. Copy it and paste it in your browser.
