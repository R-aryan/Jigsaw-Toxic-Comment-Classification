# Jigsaw Toxic Comment Classification
 Identify and classify toxic online comments

- End to End NLP Multi label Classification problem
- The Kaggle dataset can be found Here [Click Here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
 
## Steps to Run the Project:
- create a virtual environment and install requirements.txt
  
### For Training
- After Setting up the environment go to **backend/services/toxic_comment_jigsaw/application/ai/training/** and run **main.py** and the training will start.
- After training is complete the weights of the model will be saved in weights directory, and this weights can be used for inference.
  
### For Prediction/Inference
- After setting up the environment: go to **backend/services/toxic_comment_jigsaw/api** and run **app.py**.
- After running the above step the server will start.  
- You can send the POST request at this URL - **localhost:8080/toxic_comment/api/v1/predict** (you can find the declaration of endpoint under **backend/services/toxic_comment_jigsaw/api/__init__.py** )

Following are the screenshots for the output, and the request.

- Request sample 
![Sample request](https://github.com/R-aryan/Jigsaw-Toxic-Comment-Classification/tree/develop/msc/toxic_request.png)
  <br>
  <br>
- Response Sample
![Sample response](https://github.com/R-aryan/Jigsaw-Toxic-Comment-Classification/tree/develop/msc/toxic_response.png)
