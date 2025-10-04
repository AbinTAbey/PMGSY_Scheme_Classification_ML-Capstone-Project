# Intelligent Classification of PMGSY Rural Infrastructure Projects using Machine Learning - Capstone Project

## Overview

This project aims to develop a machine learning solution that automatically classifies rural infrastructure projects—such as roads and bridges—under the appropriate phase or scheme of the Pradhan Mantri Gram Sadak Yojana (PMGSY). By leveraging government-provided datasets from the AI Kosh initiative and implementing the entire workflow on IBM Cloud Lite services, the solution intends to help government agencies, infrastructure planners, and policy analysts streamline the monitoring and management of thousands of projects. Efficient classification is crucial for transparent budgeting, efficient resource allocation, and data-driven assessment of the long-term impact of various PMGSY schemes.

## Problem Statement

The Pradhan Mantri Gram Sadak Yojana (PMGSY) is a flagship rural development program by the Government of India that provides all-weather road connectivity to unconnected rural habitations. Over several phases (including PMGSY-I, PMGSY-II, and RCPLWEA), each scheme brings unique objectives, funding mechanisms, and project specifications. Manually categorizing the vast number of completed and ongoing rural infrastructure projects is labor-intensive, potentially error-prone, and difficult to scale as the volume of data grows. The challenge is to design, build, and evaluate a machine learning model that classifies each road or bridge project into its correct PMGSY_SCHEME based on its physical and financial characteristics. The goal is to offer an accurate, scalable, and reproducible system that enables government stakeholders to quickly and reliably categorize projects, improving operational efficiency and transparency in rural infrastructure development.

## Solution Overview

A **multi-class classification model** was developed to predict the correct `PMGSY_SCHEME `(such as PMGSY-I, PMGSY-II, RCPLWEA, etc.) for rural infrastructure projects based on their physical and financial attributes. Leveraging the AI Kosh dataset and fully utilizing IBM Watson Studio’s AutoAI for the entire machine learning pipeline, the solution enables end-to-end automation—from initial data preparation to deployment of a scalable evaluation system. The final trained model is published as a REST API endpoint on IBM Cloud, making it easy to integrate with existing government IT systems or project management dashboards.

Key Components:

Data Collection: Aggregated data of rural road and bridge projects sourced from the AI Kosh portal, including features such as project length, estimated cost, funding details, and completion status.

Automated ML Pipeline: The entire machine learning process—data cleaning, feature engineering, model selection, and hyperparameter tuning—was automated using IBM Watson Studio’s AutoAI.

Model Deployment: The best-performing classification model is deployed to IBM Watson Machine Learning as a secure and scalable online API endpoint.

Prediction: The deployed model receives new project data (physical and financial characteristics) and returns the predicted PMGSY scheme, enabling rapid, consistent, and transparent project categorization.
## System Architecture

![image alt](https://github.com/AbinTAbey/PMGSY_Scheme_Classification_ML-Capstone-Project/blob/84a00a8d0d1e6e7d041e035c19c3f26361adef0b/src/System_Architecture.png)

## 📊 Data Source

The dataset used for this project is:  
**Pradhan Mantri Gram Sadak Yojana (PMGSY) Project Dataset**

- **Source:** AI Kosh, Government of India  
- **Link:** [Pradhan Mantri Gram Sadak Yojana (PMGSY) Dataset](https://aikosh.indiaai.gov.in/web/datasets/details/pradhan_mantri_gram_sadak_yojna_pmgsy.html)  
- **File in this repository:** `PMGSY_DATASET.csv`  

![image alt](https://github.com/AbinTAbey/PMGSY_Scheme_Classification_ML-Project/blob/eba57139fb95829fb5a628d2563edd1730694d56/src/dataset.PNG)

**🔗 Live Interactive Dashboard:** [View Dashboard](https://claude.ai/public/artifacts/dfedc4aa-4547-4b76-942a-af8a1536a85a)

![image alt](https://github.com/AbinTAbey/PMGSY_Scheme_Classification_ML-Project/blob/550e2d5389a9c119c5a6a4c64c550a33c1da204c/src/cid_2.PNG)

### 📌 Note on Data Granularity
The provided dataset contains *aggregated project-level data* at the state and district level (e.g., number and length of road and bridge works sanctioned, associated costs, completion status, and expenditures, all linked to a specific scheme). While the developed machine learning model accurately classifies individual projects into their corresponding PMGSY scheme using these features, Future enhancements could involve integrating more granular (e.g., individual work-level or real-time) data to further refine scheme predictions and enable even more targeted analysis.


## Technologies Used

* **IBM Cloud:** The foundational cloud platform.
* **IBM Cloud Object Storage:** For secure storage of the dataset.
* **IBM Watson Studio:** The integrated environment for data science and machine learning workflows.
* **AutoAI (within Watson Studio):** Automated machine learning tool for building, training, and optimizing ML models.
* **IBM Watson Machine Learning:** Service for managing runtime environments and deploying models.
* **Git & GitHub:** For version control and repository hosting.


| Component | Technology | Purpose |
|-----------|------------|---------|
| **ML Platform** | IBM watsonx.ai Studio | Automated model development |
| **Algorithm** | XGBoost Classifier | Optimal classification performance |
| **Data Source** | AI Kosh PMGSY Dataset | Official government project data |
| **Deployment** | IBM Cloud | Scalable production environment |
| **API** | REST Endpoints | Real-time integration capability |

## Repository Contents

* `PMGSY_DATASET.csv`: The raw dataset used for training the model.
* `PMGSY_RURALAutoAI-Notebook.ipynb`: A Jupyter Notebook automatically generated by IBM Watson Studio's AutoAI, containing the code for the best-performing machine learning pipeline. This notebook provides insights into the preprocessing steps and the model architecture.

## How to Use/Run (Conceptual)

This project was primarily developed and deployed on IBM Cloud. To understand or reproduce the core steps:

1.  **IBM Cloud Account:** Access to an IBM Cloud account (Lite tier is sufficient for this project).
2.  **Provision Services:** Provision IBM Cloud Object Storage, IBM Watson Studio, and IBM Watson Machine Learning services.
3.  **Create Project:** Set up a new project in IBM Watson Studio and link it to your Cloud Object Storage.
4.  **Upload Data:** Upload `PMGSY_DATASET.csv` to your project's data assets.
5.  **Run AutoAI Experiment:** Create a new AutoAI experiment, select `PMGSY_DATASET.csv` as the data source, and choose `PMGSY_SCHEME` as the target column for multi-class classification. Run the experiment.
6.  **Save & Deploy Model:** Save the best-performing pipeline as a model. Create a Deployment Space, promote the model to this space, and then create an "Online Deployment" to get a REST API endpoint.
7.  **Test Predictions:** Use the "Test" tab in the deployed model's interface (or its API endpoint) to send new data in JSON format and receive predictions.

The `PMGSY_RURALAutoAI-Notebook.ipynb` provides the programmatic details of the generated model and can be run within a Watson Studio notebook environment.

## Model Performance

The AutoAI experiment was optimized for Accuracy, achieving **above 90% accuracy** resulting in high performance when classifying the `PMGSY_SCHEME` target based on the state, district, and associated project-level features. The model demonstrates robust ability to distinguish between different PMGSY schemes (such as PMGSY-I, PMGSY-II, RCPLWEA, etc.) using the available physical and financial attributes. Detailed performance metrics, including the confusion matrix and feature importance, can be found within the IBM Watson Studio AutoAI experiment results.
### Training of Model
![image alt](https://github.com/AbinTAbey/PMGSY_Scheme_Classification_ML-Project/blob/3772339a61a6cc538075b6a22ce73a4e4000ef69/src/Training.png)

### Accuracy of Model - XGB Classifier
![image alt](https://github.com/AbinTAbey/PMGSY_Scheme_Classification_ML-Project/blob/5e87147ed64ff878d899a0ad013b9eb9e70be3ab/src/accuracy.PNG)
## Test of Model 
![image alt](https://github.com/AbinTAbey/PMGSY_Scheme_Classification_ML-Project/blob/85403e2026a0f14434aebf42ea28faf3545256e8/src/TestPrediction.png)
### Prediction Result

![image alt](https://github.com/AbinTAbey/PMGSY_Scheme_Classification_ML-Project/blob/b19419ab178b18f53184ace3ee7a50455321c3e3/src/Model_Result%2Boutput.png) 

### Performance Metrics
| Metric | Value | Details |
|--------|-------|---------|
| **Accuracy** | 92.4% | Cross-validation performance |
| **Training Time** | 3 minutes | Complete pipeline generation |
| **Models Tested** | 8 pipelines | AutoAI comparison and selection |
| **Best Algorithm** | XGBoost | Pipeline 8 with optimization |

## Application Scenarios

### Primary Users

- **Government Officials**
  - Instantly categorize projects for faster reporting and workflow.
  - View real-time project status via integrated dashboards for better oversight.
  - Verify policy compliance for each project using automated checks.
- **Infrastructure Planners**
  - Optimize how resources are distributed across regions and schemes.
  - Manage diverse project portfolios for more strategic decision-making.
  - Use classified data to support long-term planning and forecasting.
- **Audit Teams**
  - Automate compliance reviews for government standards and procedures.
  - Quickly validate historical project classifications for transparency and audit purposes.
  - Generate reports to improve accountability and track outcomes.

### Integration Scenarios

- **ERP Systems**: Seamlessly connect the classification API to government enterprise resource platforms for live data updates.
- **Mobile Apps**: Enable real-time project verification and scheme assignment for field staff using mobile interfaces.
- **Dashboards**: Empower stakeholders with interactive, real-time analytics and project monitoring.
- **Batch Processing**: Efficiently process and reclassify large datasets of historical projects for analysis and compliance audits.

## Future Enhancements

* **Integrate Additional Data Sources:** Enhance the model’s predictive accuracy and robustness by incorporating diverse data streams, such as real-time project progress updates, satellite imagery for on-ground construction verification, and local economic indicators. This multi-source integration will provide a more comprehensive profile of each project for smarter scheme assignment.
* **Expand to Predictive Analytics::** Extend the system’s capability beyond mere scheme classification to include forecasting important outcomes. Potential enhancements include predicting the likelihood of project delays, identifying projects at risk for budget overruns, and anticipating future maintenance requirements. These predictive analytics will empower stakeholders with forward-looking insights for proactive rural infrastructure management.
* **Cover Other Government Schemes:** Broaden the model’s framework to automatically classify and monitor projects associated with other major government infrastructure initiatives. This scalable approach would facilitate the development of a unified, intelligent platform for tracking and optimizing national development programs across multiple sectors.
* **Model Monitoring & Automated Retraining:** Establish continuous performance monitoring of the deployed model to detect drift or decreases in accuracy, and automate retraining pipelines using new incoming project data, ensuring that predictions remain robust over time.
* **Explainable AI (XAI):** Integrate explainable AI techniques to provide clear, actionable insights into why a particular scheme was predicted for a project. This will enhance transparency and build trust among government stakeholders by highlighting the most influential factors in each classification decision.


## 📚 Documentation & Resources

### IBM watsonx Resources
- [IBM watsonx AutoAI Documentation](https://cloud.ibm.com/docs/watsonx-ai)
- [Watson Machine Learning Guide](https://cloud.ibm.com/docs/watson-machine-learning)
- [AutoAI Tutorials](https://developer.ibm.com/tutorials/autoai-overview/)

### Government Data Sources
- [PMGSY Dataset - AI Kosh Portal](https://aikosh.indiaai.gov.in/web/datasets/details/pradhan_mantri_gram_sadak_yojna_pmgsy.html)
- [PMGSY Official Guidelines](https://pmgsy.nic.in/)
- [AI Kosh Platform](https://aikosh.gov.in/)

### Research References
- **AutoML for Government**: S. Sharma, P. Gupta. *International Journal of AI in Public Administration*, 2023
- **XGBoost Classification**: T. Chen, C. Guestrin. *Proceedings of ACM SIGKDD*, 2016
- **Rural Infrastructure ML**: R. Kumar, A. Patel, M. Singh. *IEEE Smart Governance Conference*, 2022

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ministry of Rural Development, Government of India** for PMGSY program data
- **AI Kosh Initiative** for providing open government datasets
- **IBM watsonx Team** for AutoAI platform capabilities
- **Digital India Mission** for promoting AI in governance
  
**⭐ Star this repository if you find it helpful!**

*This project demonstrates the practical application of AI in government operations, transforming manual processes into intelligent, automated solutions for better rural infrastructure management.*
<div align="center">
  <h1>🐦 Twitter Sentiment Analysis</h1>
  <h3>Machine Learning Model for Classifying Tweet Sentiments</h3>
  <p>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python"></a>
    <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-1.0-orange.svg" alt="scikit-learn"></a>
    <a href="https://www.nltk.org/"><img src="https://img.shields.io/badge/NLTK-3.8-green.svg" alt="NLTK"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
  </p>
  <p>
    <a href="#overview">Overview</a> • 
    <a href="#features">Features</a> • 
    <a href="#installation">Installation</a> • 
    <a href="#usage">Usage</a> • 
    <a href="#results">Results</a>
  </p>
</div>
<hr>
<h2 id="overview">📖 Overview</h2>
<p>
A comprehensive <strong>Natural Language Processing</strong> project that classifies Twitter sentiment using the <strong>Sentiment140 dataset</strong> containing <strong>1.6 million tweets</strong>. The model uses logistic regression to predict whether tweets express positive or negative sentiment with <strong>77.79% accuracy</strong>.
</p>
<h3>🎯 Key Highlights</h3>
<ul>
  <li>🔍 Processes 1.6M tweets from Sentiment140 dataset</li>
  <li>🧹 Advanced text preprocessing with stemming & stopword removal</li>
  <li>🤖 Logistic Regression classifier with TF-IDF vectorization</li>
  <li>⚡ Achieves 77.79% test accuracy</li>
  <li>📊 Balanced binary classification (Positive/Negative)</li>
</ul>
<hr>
<h2 id="features">✨ Features</h2>
<ul>
  <li><strong>Automated Data Collection</strong>: Kaggle API integration for seamless dataset download</li>
  <li><strong>Text Preprocessing Pipeline</strong>: Comprehensive cleaning including URL removal, stemming, and stopword filtering</li>
  <li><strong>TF-IDF Vectorization</strong>: Converts text into meaningful numerical features</li>
  <li><strong>Machine Learning Model</strong>: Logistic Regression optimized for sentiment classification</li>
  <li><strong>Performance Metrics</strong>: Detailed accuracy evaluation on train/test splits</li>
</ul>
<hr>
<h2 id="tech-stack">🛠️ Tech Stack</h2>
<table>
  <tr><th>Category</th><th>Technologies</th></tr>
  <tr><td>Language</td><td>Python 3.11+</td></tr>
  <tr><td>ML Framework</td><td>scikit-learn</td></tr>
  <tr><td>NLP</td><td>NLTK, RegEx</td></tr>
  <tr><td>Data Processing</td><td>NumPy, Pandas</td></tr>
  <tr><td>Platform</td><td>Google Colab / Jupyter Notebook</td></tr>
</table>
<hr>
<h2 id="dataset">📊 Dataset</h2>
<p><strong>Source:</strong> <a href="https://www.kaggle.com/datasets/kazanova/sentiment140">Sentiment140 Dataset</a> from Kaggle</p>
<table>
  <tr><th>Feature</th><th>Description</th></tr>
  <tr><td>Total Tweets</td><td>1,599,999</td></tr>
  <tr><td>Classes</td><td>Binary (0: Negative, 4: Positive)</td></tr>
  <tr><td>Features</td><td>Target, IDs, Date, Flag, User, Text</td></tr>
  <tr><td>Distribution</td><td>800,000 negative + 800,000 positive</td></tr>
</table>
<hr>
<h2 id="installation">🚀 Installation</h2>
<h3>Prerequisites</h3>
<ul>
  <li>Python 3.11+</li>
  <li>Kaggle Account & API Token</li>
</ul>
<h3>Setup</h3>
<pre><code>git clone https://github.com/yourusername/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
pip install numpy pandas scikit-learn nltk kaggle
</code></pre>
<pre><code>import nltk
nltk.download('stopwords')
</code></pre>
<pre><code>mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
</code></pre>
<hr>
<h2 id="usage">💻 Usage</h2>
<h3>Running the Notebook</h3>
<pre><code>jupyter notebook Twitter_Sentiment_analysis-1.ipynb</code></pre>
<h3>Quick Start Code</h3>
<pre><code># Import libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Load preprocessed data
twitter_data = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1')

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
</code></pre>
<hr>
<h2 id="preprocessing">🔧 Preprocessing Pipeline</h2>
<h3>Text Cleaning Steps</h3>
<ol>
  <li>Remove Special Characters: Eliminates @mentions, URLs, and punctuation</li>
  <li>Lowercase Conversion: Standardizes text format</li>
  <li>Stopword Removal: Filters common words (the, is, at, etc.)</li>
  <li>Stemming: Reduces words to root form using Porter Stemmer</li>
  <li>Vectorization: TF-IDF transformation for numerical representation</li>
</ol>
<h3>Example Transformation</h3>
<pre><code>Original:  "@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer!"
Processed: "switchfoot awww bummer"
</code></pre>
<hr>
<h2 id="results">📈 Results</h2>
<h3>Model Performance</h3>
<table>
  <tr><th>Metric</th><th>Score</th></tr>
  <tr><td>Training Accuracy</td><td>77.88%</td></tr>
  <tr><td>Test Accuracy</td><td>77.79%</td></tr>
  <tr><td>Model</td><td>Logistic Regression</td></tr>
  <tr><td>Iterations</td><td>1000</td></tr>
</table>
<ul>
  <li>Consistent train/test accuracy indicates no overfitting</li>
  <li>Balanced performance across positive and negative classes</li>
  <li>Efficient processing of 1.6M+ tweets</li>
</ul>
<hr>
<h2 id="structure">📁 Project Structure</h2>
<pre><code>twitter-sentiment-analysis/
│
├── Twitter_Sentiment_analysis-1.ipynb  # Main implementation notebook
├── README.md                            # Project documentation
├── kaggle.json                          # Kaggle API credentials
├── requirements.txt                     # Python dependencies
└── data/
    └── sentiment140.zip                 # Downloaded dataset
</code></pre>
<hr>
<h2 id="use-cases">🎯 Use Cases</h2>
<ul>
  <li>Social Media Monitoring: Track brand sentiment in real-time</li>
  <li>Customer Feedback: Analyze product reviews and opinions</li>
  <li>Market Research: Understand public perception of topics</li>
  <li>Campaign Analysis: Measure marketing campaign effectiveness</li>
  <li>Political Analysis: Gauge public opinion on policies</li>
</ul>
<hr>
<h2 id="future">🔮 Future Enhancements</h2>
<ul>
  <li>Implement deep learning models (LSTM, BERT, Transformers)</li>
  <li>Add multi-class sentiment (Positive, Negative, Neutral)</li>
  <li>Create real-time Twitter API integration</li>
  <li>Build interactive web dashboard with Streamlit</li>
  <li>Deploy model as REST API using Flask/FastAPI</li>
  <li>Add emoji and emoticon sentiment analysis</li>
  <li>Implement cross-validation and hyperparameter tuning</li>
</ul>
<hr>
<h2 id="contributing">🤝 Contributing</h2>
<ol>
  <li>Fork the repository</li>
  <li>Create a feature branch (<code>git checkout -b feature/AmazingFeature</code>)</li>
  <li>Commit changes (<code>git commit -m 'Add AmazingFeature'</code>)</li>
  <li>Push to branch (<code>git push origin feature/AmazingFeature</code>)</li>
  <li>Open a Pull Request</li>
</ol>
<hr>
<h2 id="license">📝 License</h2>
<p>
  This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.<br>
  The Sentiment140 dataset is available under Kaggle's terms of use.
</p>
<hr>
<h2 id="author">👤 Author</h2>
<p>
  <strong>Your Name</strong><br>
  GitHub: <a href="https://github.com/yourusername">@yourusername</a><br>
  LinkedIn: <a href="https://linkedin.com/in/yourprofile">Your Name</a><br>
  Portfolio: <a href="https://yourportfolio.com">yourportfolio.com</a>
</p>
<hr>
<h2 id="acknowledgments">🙏 Acknowledgments</h2>
<ul>
  <li><a href="https://www.kaggle.com/datasets/kazanova/sentiment140">Sentiment140 Dataset</a> by Kazanova</li>
  <li>Stanford University for the original dataset creation</li>
  <li>NLTK team for natural language processing tools</li>
  <li>scikit-learn community for machine learning framework</li>
</ul>
<div align="center">
  <h3>⭐ Star this repository if you find it helpful!</h3>
  <p>Made with ❤️ by Abin T Abey</p>
</div>
