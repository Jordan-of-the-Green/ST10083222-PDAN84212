# ST10083222-PDAN84212-POE-ReadMe  

# Task 1 Description:    
An end-to-end sentiment analysis app utilizing PySpark and Keras. It processes large datasets, cleans and visualizes data, and builds an LSTM model for text classification. The app enables efficient data handling, predictive modeling, and visual performance tracking, ideal for scalable and accurate sentiment analysis.

# Task 2 Description:
This task involves building and evaluating a logistic regression model to classify data using PySpark for data processing and Python for model development. The aim is to develop a binary classification model that provides an efficient and scalable approach to predicting outcomes from large datasets.

# Task 3 Description:
This project is an image classification application that leverages TensorFlow to classify images as either "REAL" or "FAKE." It processes the CIFAR-10 dataset, applies data augmentation techniques, and utilizes a custom-built Convolutional Neural Network (CNN) for binary classification. The project efficiently handles large datasets, augments data for better model generalization, and visualizes the model’s performance during training.

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
* For downloading a project code from a repository, navigate to the GitHub repository link provided by the project team.  
* Clone the repository using the Git extension integrated into Visual Studio Code or download the code as a zip file.  

Step 7  
* Once downloaded, open Visual Studio Code, click on "File" > "Open Folder," and navigate to the folder containing the project code.  
* You are now ready to use Visual Studio Code for your project development.

# How to Install the Dataset Part 1:  
Dataset ~  
Step 1:  
* Using the link for the 'Dataset Reference' shown in the 'Theory' section of the assignment will take you to the correct site.
![image](https://github.com/user-attachments/assets/b06fd1ea-f876-4ad8-9927-763b31b86011)
Step 2:
* Scroll down and click the download icon as shown in the guide below:  
![image](https://github.com/user-attachments/assets/fce541d7-a534-4ca2-aed6-5b5f1c98c07d)
Step 3:
* Copy and paste the CSV file into your work, and the code should be functional now.  

# How to Install the Dataset Part 2:  
Dataset ~  
Step 1:  
* Using the link for the 'Dataset Reference' shown in the 'Theory' section of the assignment will take you to the correct site.
![image](https://github.com/user-attachments/assets/cd2dedb7-a9c8-4683-bca7-117151a65862)
Step 2:
* Scroll down and click the download icon as shown in the guide below:  
![image](https://github.com/user-attachments/assets/b8e95772-70d4-479b-878d-4e9ffc0df883)
Step 3:
* Copy and paste the CSV file into your work, and change the CSV name to 'Jordan Green. ST10083222. PDAN8412. POE 2'; the code should be functional now.  

# How to Install the Dataset Part 3:  
Dataset ~  
Step 1:  
* Using the link for the 'Dataset Reference' shown in the 'Theory' section of the assignment will take you to the correct site.
![image](https://github.com/user-attachments/assets/ddf0b386-f4e7-4a45-b476-fff977454db0)
Step 2:
* Scroll down and click the download icon as shown in the guide below:  
![image](https://github.com/user-attachments/assets/37c8e7db-03df-4fcc-9c75-d8e7847c0983)
Step 3:
* Copy and paste the CSV file into your work, and the code should be functional now. 

# Task 1
## Key Features:  
### Model Training and Prediction:  
• Train an LSTM (Long Short-Term Memory) model using Keras to classify text sentiment based on target values.  
• Tokenize text data and pad sequences for input to the LSTM model, ensuring consistent input length.  
• Compile the model with an 'adam' optimizer and binary cross-entropy loss function, and evaluate performance using accuracy metrics.  
• Implement early stopping during training to prevent overfitting and ensure optimal performance on the validation set.  

### Data Preprocessing:  
• Load and sample large datasets efficiently using PySpark for scalable data processing.  
• Clean data by selecting necessary columns, dropping missing values, and removing duplicates.  
• Convert PySpark DataFrames to Pandas DataFrames for compatibility with Keras preprocessing tools.  
• Tokenize text data using Keras Tokenizer and convert it into padded sequences for LSTM input.    

### Data Visualization:  
• Visualize missing data in the dataset using a heatmap, ensuring no null values exist.  
• Plot model performance metrics, including accuracy and loss over epochs, for both training and validation sets.    

### User-friendly Interface:  
• Utilize PySpark for efficient handling of large datasets and seamless transitions to Keras for deep learning tasks.  
• Provide clear data cleaning steps and a visual representation of model performance during the training process.     

## Why Use This App?
• Sentiment Analysis: Classify text sentiment using LSTM-based deep learning, providing powerful insights from text data.  
• Scalable Processing: Leverage PySpark for handling large datasets efficiently, ensuring performance even with substantial data.  
• Easy Model Visualization: Track and understand model performance through intuitive accuracy and loss graphs during training.  
• Practical Learning: A hands-on approach to building deep learning models with practical insights into data preprocessing and model evaluation.    

# Error features:     
- **No Major Errors**: All components of the system have been thoroughly tested and work as expected.

# Task 2
## Key Features:
### Logistic Regression Model:  
• Train a logistic regression model for binary classification on a large dataset.  
• Utilize gradient descent to optimize the weights and bias for logistic regression.  
• Implement sigmoid activation for non-linearity and binary output.  
• Train the model iteratively using training data, and validate performance on unseen data.  

### Data Preprocessing:  
• Efficient data wrangling using PySpark for large datasets, cleaning missing values, and dropping irrelevant columns.  
• Convert PySpark DataFrames into Pandas DataFrames for compatibility with Python libraries.  
• Label encoding for categorical data to ensure compatibility with logistic regression.  

### Data Visualization:  
• Generate visualizations such as count plots and heatmaps to understand data distribution and clean missing values.  
• Visualize the correlation matrix using background gradients to identify relationships between variables.  

### Model Evaluation:  
• Calculate key evaluation metrics: accuracy, F1 score, precision, and recall.  
• Display a detailed classification report with per-class performance metrics.  
• Confusion matrix visualization for understanding model predictions.  

## Why Use This Model?  
• Logistic regression is simple yet effective for binary classification tasks.  
• Suitable for large datasets due to efficient PySpark handling of data.  
• Provides clear insights into model performance using visualizations and classification reports.  
• Easily extendable for further tuning and deployment.  

# Error features:  
- **No Major Errors**: No major issues were encountered during development, but overfitting occurred with the imbalanced dataset.  

# Task 3
## Key Features:
### Model Training and Prediction:
- Build a custom Convolutional Neural Network (CNN) using TensorFlow/Keras for binary classification of images (REAL vs. FAKE).
- The model consists of convolutional layers, max pooling, and dropout for regularization to prevent overfitting.
- The model is compiled with the Adam optimizer, binary cross-entropy loss function, and accuracy as the evaluation metric.
- The model is trained on the processed dataset and used to predict the class of unseen images.

### Data Preprocessing and Augmentation:
- Use TensorFlow's ImageDataGenerator to load and preprocess the dataset with augmentation techniques like horizontal flipping, zoom, and rotation.
- Reduce the dataset size by limiting the number of images per class to improve training performance and reduce memory usage.
- Preprocess the images by resizing them and normalizing pixel values to the [0, 1] range for improved model efficiency.

### Visualization and Evaluation:
- Visualize model training performance with graphs for accuracy and loss over epochs.
- After training, evaluate the model's performance using a confusion matrix, classification report, and other evaluation metrics like precision, recall, and F1-score.
- Use the evaluation results to identify potential areas of improvement for the model.

### Error Handling and Debugging:
- Handle missing or incorrectly formatted data by cleaning the dataset, and ensuring the images are correctly labeled and organized.
- No major errors have been found during testing. If any issues arise, debugging is facilitated by the clear output of model performance metrics (accuracy, loss, confusion matrix).

## Why Use This App?
- **Image Classification**: Automatically classify images as "REAL" or "FAKE" using a deep learning-based CNN model.
- **Scalable Dataset Handling**: Efficiently process large datasets through TensorFlow's ImageDataGenerator and augmentation strategies.
- **Model Evaluation**: Track model performance with visualization tools and evaluation metrics like confusion matrix and classification reports.
- **Practical Learning**: Gain hands-on experience in building, training, and evaluating deep learning models for image classification.

## Limitations and Future Improvements:
- The current model might require further fine-tuning for improved accuracy, especially when working with larger or more complex datasets.
- Future improvements could involve using pre-trained models like ResNet or Inception for better feature extraction and model performance.
- Implementing more sophisticated data augmentation strategies could further enhance model robustness.

## Error Features:
- **No Major Errors**: All components of the system have been thoroughly tested and work as expected.
