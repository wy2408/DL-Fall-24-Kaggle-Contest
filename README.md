# DL-Fall-24-Kaggle-Contest
NYU Tandon Deep Learning Midterm Project
In this competition, participants are tasked with the Supervised-Fine Tuning(SFT) of Llama3-8B model to predict the correctness of answers to math questions. The goal is to assess whether the provided answer to each question is correct or not. We will host this competition on Kaggle. You must join the hosted competition on Kaggle and make a submission. You can find a detailed description of the contest on the overview page on Kaggle. We hope you enjoy this contest and have a great learning experience. All the best!
Competition Rules
Model: Only Llama3-8B model is permitted
Dataset: NO external dataset is permitted. You must use the given training dataset. You can NOT use the test partition for any kind of training.
Max team size: 3
Every participant of your team should register on Kaggle with their NYU NetID email address and join the contest and the respective team.
NYU's academic rules and Kaggle's contest rules hold true.

Joining the contest
Register your team here: DL Fall 24 team registrations	
Create an account on Kaggle using your NYU email id(NetId) and use this link to join the contest.
Make sure to form a team and join your teammates on Kaggle. 
Deadlines
Contest Deadline: EST - 11:59 PM, Nov 15, 2024
Report submission deadline on the Gradescope: EST - 11:59 PM,  Nov 17, 2024
Grading rubric
Accuracy  [50/100]
< 0.6 : 0  to 25
0.6 - Baseline(0.756) : 25 to 35
Baseline(0.756) - 0.80: 35 to 40 
0.80 - 0.85 : 40 to  50 > 0.85 : 50

Report [35/100]
Code [15/100]
Create a Github repository and upload related code. Include a link in your report.
Share a link to your model's weight by uploading on cloud storage.  
Training, Validation, Inference code quality and reproducibility.
Extra 20 points to First three ranks and 10 extra points to fourth to sixth ranks on the final Kaggle leaderboard. 
PS: Half of the test cases are hidden. Kaggle will show scores on only 50% of the test cases before the contest ends. Once the contest ends it will show the final leaderboard. 



Submission Instructions
Kaggle Contest
You can do 5 submissions per day per team on the Kaggle. You must do it at least once throughout the contest. You need to submit a CSV file with two columns ‘ID’ and ‘is_correct’. More details and a sample submission file is available on the contest page.
Report
You must submit a report in ACL format on the Gradescope. Overleaf template:  ACL 2023 Proceedings Template - Overleaf, Online LaTeX Editor . 
Make sure you have Introduction, Dataset, Model description, Experimentation description and hyperparameters settings, Results and Conclusion sections. What methods you tried, what worked and what didn’t. 
Include a link to your training/inference notebook and the model weights.
Getting Started
Starter Notebook:
 NYU-DL-24-Contest_starter.ipynb
Ideas for improvement on the baseline:
Read more about LoRA and adjust LoRA parameters and Quantization strategies. 
Hyperparameter tuning.
Try different prompts.
Make use of the  ‘Solution’ column in your training as they can impart a sense of reasoning to your model. 
