# Timeline-based-List-Question-Answering

## 1. Initial (one-time) Setup

Follow these steps to prepare your environment for the project:

### 1.1 Set up WSL (for Windows only)

If you don't already have WSL, run
```bash
wsl --install
```

Otherwise:
```bash
wsl
```

### 1.2 Navigate to the Project Directory

Change to the project's root directory:

```bash
cd <.../Timeline-based-List-Question-Answering>
```

### 1.3 Set Up a Virtual Environment

Create an isolated environment for your project:

1. **Create a virtual environment:**

```bash
python3 -m venv venv
```

2. **Activate the virtual environment:**
```bash
source venv/bin/activate
```

### 1.4 Install Required Packages

Install the necessary packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 2. Running after the initial setup
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
