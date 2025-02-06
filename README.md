# Self-analysis-mental-health-

Mental health disorders impact millions globally, and early identification of symptoms is crucial in seeking appropriate support. This project focuses on developing an AI-driven Mental Health Severity Prediction System that classifies users into three categories: Mild, Moderate, and Severe.
System Overview: The core of the system is a machine learning model that evaluates users' mental health based on their inputs. Additionally, it incorporates a mental health chatbot, powered by Google Gemini, to provide users with explanations and coping mechanisms, ensuring a supportive and educational interaction.
User Interface: An interactive and user-friendly interface has been developed using Stream lit, enhancing the overall user experience and making the system accessible and easy to navigate
Technologies used:
1.	Python
2.	LLM – Google Gemini 1.5 flash
3.	Scikit-learn
4.	Stream lit and Job lib
5.	Matplotlib and Seaborn


Dataset and Preprocessing Steps
Dataset Used
We utilized the Mental Health in Tech Survey dataset, which comprises responses from individuals working in tech companies. This dataset includes various features related to the work environment, mental health history, age, and accessibility to treatment.
Data Cleaning & Preprocessing Steps
Handling Missing Values: Imputed missing values using mode/mean substitution to ensure data completeness.
Encoding Categorical Variables:
• Label Encoding: Applied to binary categorical features.
• One-Hot Encoding: Utilized for multi-class categorical features to convert them into numerical values.

Feature Selection: Identified and selected 23 important features based on their correlation with mental health severity levels.
Train-Test Split: Split the dataset into 80% training data and 20% testing data to evaluate the model's performance accurately.

Model selection and rationale
Model Training and Evaluation
Model Training: Four machine learning algorithms were employed to train the Mental Health Severity Prediction System: Random Forest, Artificial Neural Network (ANN), XG Boost, and Logistic Regression. Each model was trained on the pre-processed dataset, and their performances were evaluated based on various metrics such as accuracy, precision, recall, and F1-score.
1. Random Forest (Best Model): The Random Forest classifier was the top performer in terms of accuracy and overall performance. It uses an ensemble of decision trees to improve prediction accuracy and control over-fitting.
ROC-AUC for the Best Model: The ROC-AUC (Receiver Operating Characteristic - Area Under Curve) metric provides an aggregate measure of performance across all classification thresholds. For the Random Forest model, the ROC-AUC score was 0.79 indicating its superior capability to distinguish between the different severity levels.


Confusion Matrix for the Best Model: The confusion matrix for the Random Forest model provides a detailed breakdown of the model's predictions compared to the actual classifications. It shows the True Positives, True Negatives, False Positives, and False Negatives, allowing us to understand the model's accuracy in predicting each category.



SHAP Implementation:
To interpret the model's decisions, I utilized SHAP (Shapley Additive explanations) to understand the contribution of each feature to the model's predictions. SHAP analysis revealed that features such as Family history, Age, work culture and Social Support played a crucial role in determining mental health outcomes. By visualizing these SHAP values, I gained valuable insights into the factors influencing mental health, enabling the identification of key areas for intervention and providing transparency in the model’s decision-making process.
So, for our best performed model “Random Forest” with an accuracy of 77% & f1 of 78% has SHAP:


How to run the inference script
The inference script is designed to load the trained model, process user-provided inputs, and predict the mental health severity level (Mild, Moderate, Severe). The script ensures that predictions are made accurately using the Random Forest model (‘mental_health_model.pkl’), which was found to be the most effective model during evaluation. The user needs to input features such as age, gender, history of mental health issues, work environment factors, and access to mental health resources.
After the input is provided, the script processes the data to match the same preprocessing steps used during training. This ensures that the model receives data in a format it understands. The Random Forest model then makes a prediction, classifying the user’s mental health severity into one of the three categories:
✔ Mild – Indicating minimal signs of mental health concerns. ✔ Moderate – Suggesting noticeable mental health struggles that may require intervention. ✔ Severe – Indicating significant mental health concerns that may need professional help.
The predicted severity level is displayed, and also the AI chatbot is integrated, it generates a natural language explanation for the user. The chatbot also provides personalized coping mechanisms based on the severity level. The inference script is designed to be lightweight and efficient, ensuring that predictions are generated instantly when user data is provided. It can be run as a standalone script or integrated into a web application (Streamlit) for a more interactive experience.
Let’s discuss about the chatbot we developed for this project!


UI/CLI usage instructions
Web App
The web app features mental health severity prediction using the Logistic Regression model, and an AI chatbot powered by Gemini 1.5 Flash. Users can input their symptoms in the sidebar, and the model predicts their mental health severity (Mild, Moderate, Severe). The AI chatbot provides explanations and coping strategies based on the results.


LLM Experimentation --- AI Chatbot (Gemini 1.5 Flash)
The AI chatbot, integrated into the mental health severity prediction web app, utilizes Gemini 1.5 Flash to provide contextual explanations and coping strategies based on user queries. It enhances the prediction system by offering natural language responses tailored to the predicted severity level (Mild, Moderate, Severe).
How the Chatbot Works
• Users can ask mental health-related questions inside the web app.
• Gemini 1.5 Flash API provides natural language explanations & coping strategies.
Example Question:
"How do I deal with stress at work?"
AI Response: "Managing stress at work involves mindfulness exercises, setting boundaries, and seeking support from supervisors or profe


Breakdown of the work:





Drawbacks and Edge Cases:
1.
If the dataset is imbalanced (e.g., more "Mild" cases than "Severe"), the model may favor common categories and struggle with rare cases. The model is trained on a specific dataset (Mental Health in Tech Survey), meaning it may not generalize well to diverse populations (e.g., non-tech industries, different cultures, or younger/older age groups).
2.
The chatbot cannot replace professional mental health advice and might provide generalized or non-contextual responses for severe cases. If a user asks "I feel extremely depressed, what should I do?", the chatbot might not detect urgency and respond with generic coping strategies instead of urging them to seek immediate help.






