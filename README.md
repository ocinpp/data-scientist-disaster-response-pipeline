# Disaster Response Pipeline Project

## Introduction

The project is divided into 3 parts.

- ETL
- Machine Learning Pipeline
- Web App

## Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database

        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

    - To run ML pipeline that trains classifier and saves

        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Deploying to AWS Beanstalk

To deploy the app to AWS Beanstalk, there are some rules that have to be followed:

- Using **application.py** as the filename and providing a callable application object (the Flask object, in this case) allows Elastic Beanstalk to easily find your application's code
- The Flask object within **application.py** must be **application**. By assigning **app** to be a reference to **application**, there is no need to rename all **app** to **application**
- **.ebextensions/python.config** can be used to specify the file that contains the WSGI application. By this way, there is no need to change the application filename to **application.py**

```yaml
option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: app/run.py
```

To view the logs, we can go to the management console, navigate to **Logs**, and choose **Request Logs**

## Screenshots

### Main Screen

![](/images/main.png)

### Distribution of Categories (Pie Chart)

![](/images/chart1.png)

### Distribution of Categories (Top 10 and Last 10)

![](/images/chart2.png)

### Message Classification (With Result)

![](/images/classify.png)

### Message Classification (No Result)

![](/images/classify-none.png)

## Discussions

Based on the categories that the ML algorithm classifies text into, the first three are **Aid Related**, **Weather Related** and **Direct Report**. So the messages should be sent to organizations that can provide medical aid and can evacuate people from extreme weather conditions.

This dataset is imbalanced (ie some labels like water have few examples). For the labels with so few examples, it means that there is not enough training data to build a good model.

Take the label **water** as an example:

| | precision | recall  | f1-score | support  |
| ----- |-----------|-----|----------|-----:|
| 0 | 0.97 | 0.99 | 0.98 | 4932 |
| 1 | 0.84 | 0.44 | 0.58 | 312  |
| avg / total | 0.96 | 0.96 | 0.96 | 5244  |

The recall rate is quite low (0.44). With low recall rate, it means many "positive" are now classified as "negative" and this may lead to missing a lot of disaster information. So, it is better to improve the recall rate so that the false negatives can be reduced.

## Demo

http://disasterresponsepipeline-env.gpmtqpkvp7.ap-northeast-1.elasticbeanstalk.com/