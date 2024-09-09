# ST10083222-PDAN84212-POE-ReadMe  

# Task 1 Description:    
An end-to-end sentiment analysis app utilizing PySpark and Keras. It processes large datasets, cleans and visualizes data, and builds an LSTM model for text classification. The app enables efficient data handling, predictive modeling, and visual performance tracking, ideal for scalable and accurate sentiment analysis.

# How to Install and Run the Project:    
Visual Studio Code Project ~  
Step 1  
* Ensure that your computer meets the system requirements to run Visual Studio Code.  
* Check for available disk space to accommodate the application.  
* Verify the necessary system requirements for your operating system.  

Step 2  
* Navigate to the official Visual Studio Code website at https://code.visualstudio.com/.  
* Download the latest version of Visual Studio Code suitable for your operating system (Windows, macOS, or Linux).  

Step 3  
* Review and agree to the terms and conditions presented during the installation process.  

Step 4  
* Once installed, launch Visual Studio Code from your applications menu or desktop shortcut.  
* Familiarize yourself with the user interface and available features.  

Step 5  
* To start coding, create a new file or open an existing project folder by selecting "File" > "Open Folder" from the menu.  
* Begin coding and exploring the various functionalities offered by Visual Studio Code.  

Step 6  
* To download a project code from a repository, navigate to the GitHub repository link provided by the project team.  
* Clone the repository using the Git extension integrated into Visual Studio Code or download the code as a zip file.  

Step 7  
* Once downloaded, open Visual Studio Code, click on "File" > "Open Folder," and navigate to the folder containing the project code.  
* You are now ready to use Visual Studio Code for your project development.

# How to Install the Dataset:  
Dataset ~  
Step 1:  
* Use the link for the 'Dataset Reference' shown in the 'Theory' section of the assignment. It will take you to the correct site where you can download the CSV file.
![image](https://github.com/user-attachments/assets/b06fd1ea-f876-4ad8-9927-763b31b86011)
Step 2:
* Scroll down and click the download icon as shown in the guide below. The download icon, not the button saying 'Download':  
![image](https://github.com/user-attachments/assets/fce541d7-a534-4ca2-aed6-5b5f1c98c07d)
Step 3:
* Copy and paste the CSV file into your work, and the code should be functional now.  

# Task 1
## Key Features:  
## Model Training and Prediction:  
• Train an LSTM (Long Short-Term Memory) model using Keras to classify text sentiment based on target values.  
• Tokenize text data and pad sequences for input to the LSTM model, ensuring consistent input length.  
• Compile the model with an 'adam' optimizer and binary cross-entropy loss function, and evaluate performance using accuracy metrics.  
• Implement early stopping during training to prevent overfitting and ensure optimal performance on the validation set.  

## Data Preprocessing:  
• Load and sample large datasets efficiently using PySpark for scalable data processing.  
• Clean data by selecting necessary columns, dropping missing values, and removing duplicates.  
• Convert PySpark DataFrames to Pandas DataFrames for compatibility with Keras preprocessing tools.  
• Tokenize text data using Keras Tokenizer and convert it into padded sequences for LSTM input.    

## Data Visualization:  
• Visualize missing data in the dataset using a heatmap, ensuring no null values exist.  
• Plot model performance metrics, including accuracy and loss over epochs, for both training and validation sets.    

## User-friendly Interface:  
• Utilize PySpark for efficient handling of large datasets and seamless transitions to Keras for deep learning tasks.  
• Provide clear data cleaning steps and a visual representation of model performance during the training process.     

## Why Use This App?
• Sentiment Analysis: Classify text sentiment using LSTM-based deep learning, providing powerful insights from text data.  
• Scalable Processing: Leverage PySpark to handle large datasets efficiently, ensuring performance even with substantial data.  
• Easy Model Visualization: Track and understand model performance through intuitive accuracy and loss graphs during training.  
• Practical Learning: A hands-on approach to building deep learning models with practical insights into data preprocessing and model evaluation.    

# Error features:     
- No issues that I could find.  
