# Intelligent Classification of PMGSY Rural Infrastructure Projects using Machine Learning
##Overview
This project aims to develop a machine learning solution that automatically classifies rural infrastructure projects—such as roads and bridges—under the appropriate phase or scheme of the Pradhan Mantri Gram Sadak Yojana (PMGSY). By leveraging government-provided datasets from the AI Kosh initiative and implementing the entire workflow on IBM Cloud Lite services, the solution intends to help government agencies, infrastructure planners, and policy analysts streamline the monitoring and management of thousands of projects. Efficient classification is crucial for transparent budgeting, efficient resource allocation, and data-driven assessment of the long-term impact of various PMGSY schemes.
##Problem Statement
The Pradhan Mantri Gram Sadak Yojana (PMGSY) is a flagship rural development program by the Government of India that provides all-weather road connectivity to unconnected rural habitations. Over several phases (including PMGSY-I, PMGSY-II, and RCPLWEA), each scheme brings unique objectives, funding mechanisms, and project specifications. Manually categorizing the vast number of completed and ongoing rural infrastructure projects is labor-intensive, potentially error-prone, and difficult to scale as the volume of data grows. The challenge is to design, build, and evaluate a machine learning model that classifies each road or bridge project into its correct PMGSY_SCHEME based on its physical and financial characteristics. The goal is to offer an accurate, scalable, and reproducible system that enables government stakeholders to quickly and reliably categorize projects, improving operational efficiency and transparency in rural infrastructure development.
Solution Overview
A multi-class classification model was developed to predict the correct PMGSY_SCHEME (such as PMGSY-I, PMGSY-II, RCPLWEA, etc.) for rural infrastructure projects based on their physical and financial attributes. Leveraging the AI Kosh dataset and fully utilizing IBM Watson Studio’s AutoAI for the entire machine learning pipeline, the solution enables end-to-end automation—from initial data preparation to deployment of a scalable evaluation system. The final trained model is published as a REST API endpoint on IBM Cloud, making it easy to integrate with existing government IT systems or project management dashboards.

Key Components:

Data Collection: Aggregated data of rural road and bridge projects sourced from the AI Kosh portal, including features such as project length, estimated cost, funding details, and completion status.

Automated ML Pipeline: The entire machine learning process—data cleaning, feature engineering, model selection, and hyperparameter tuning—was automated using IBM Watson Studio’s AutoAI.

Model Deployment: The best-performing classification model is deployed to IBM Watson Machine Learning as a secure and scalable online API endpoint.

Prediction: The deployed model receives new project data (physical and financial characteristics) and returns the predicted PMGSY scheme, enabling rapid, consistent, and transparent project categorization.


