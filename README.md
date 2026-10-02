# Flight-Delay-Prediction-System---Group-3-Project
A machine learning system that predicts flights that will be delayed by at least 15 minutes.  Flight delays affect travelers, airlines, and airports daily, causing angry travelers, frustrated airport employees, and reduced customer satisfaction with specific airlines and airports.  
An application that can help predict if a flight will be delayed will minimize the last-minute struggle to adjust flights and help travelers have a better overall experience. 
We plan to use historical flight data to identify patterns that may help predict whether a flight will be delayed. 
Our initial model will focus on predicting whether a flight will arrive at least 15 minutes late. The U.S. Department of Transportation’s Bureau of Transportation Statistics defines a delayed flight as one that arrives 15 or more minutes after its scheduled arrival time [1]​​. 
The model may later be expanded to predict the expected delay in minutes if time and data availability allow. 

<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/872ec575-39c6-4fd0-9984-b152f27f2bb2" />


# Diagram of ML system

<img width="1898" height="1034" alt="image" src="https://github.com/user-attachments/assets/b1177760-6b1b-4584-85a3-9fb0932c9aec" />

# Experiment 1 - Does expanding the training dataset from one month to three (Q1 2026) improve model performance?​

What we learned:​

More historical training data helped the Random Forest capture a much larger share of actual delays, but increased false alarms. ​

This suggests that additional training data helped identify more true delays but created a tradeoff between recall and false alarms.​

​

The next experiment should focus on improving the precision/recall tradeoff rather than simply adding more data.


# Team Members
Megan Fields · Audrey Hartford · Lila Isett · Jamie Jordan · Melissa Sierocki

