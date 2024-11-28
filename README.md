<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>AutoML Framework</h1>

<p>This project automates key steps of the machine learning pipeline, including data preprocessing, model selection, hyperparameter tuning, and evaluation. It helps streamline the development process, reducing manual effort while ensuring optimal model performance.</p>

<h2>Setup Instructions</h2>
<ol>
    <li><strong>Create and Activate Virtual Environment</strong>
        <pre><code>python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`</code></pre>
    </li>
    <li><strong>Install pip-tools</strong>: Install `pip-tools` for dependency management:
        <pre><code>pip install pip-tools</code></pre>
    </li>
    <li><strong>Compile and Sync Dependencies</strong>: Use `pip-compile` to generate the `devs.txt` file and `pip-sync` to install the dependencies:
        <pre><code>pip-compile dev.in
pip-sync dev.txt</code></pre>
    </li>
    <li><strong>Running the Project</strong>Execute the main pipeline:
        <pre><code>python main.py</code></pre>
    </li>
    
</ol>

