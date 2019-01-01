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

## Deploying to AWS Elastic Beanstalk

Create a new EC2 key pair in **EC2 management console** > **Network & ecurity** > **Key Pairs** and download the **.pem** file.

Change the permission of the **.pem** file. It is required that your private key files are NOT accessible by others.

- Remove everyone
- Add user (edit, read/run, read, write)

Create a new web server environment using a preconfigured Python platform.

![](/images/create.png)

Click **Configure more options** and then modify **Security**

In **EC2 key pair**, choose the created key pair.

![](/images/security.png)

Before deploying the app to AWS Beanstalk, there are some guidelines that have to be followed:

- Using **application.py** as the filename and providing a callable application object (the Flask object, in this case) allows Elastic Beanstalk to easily find your application's code
- The Flask object within **application.py** must be **application**. By assigning **app** to be a reference to **application**, there is no need to rename all **app** to **application**
- **.ebextensions/python.config** can be used to specify the file that contains the WSGI application. By this way, there is no need to change the application filename to **application.py**

```yaml
option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: app/run.py
```

We also have to set **AddGlobalWSGIGroupAccess** to solve the issue discussed in the article [Deploying SciPy into AWS Elastic Beanstalk](https://medium.com/@DaveJMcKeown/deploying-scipy-into-aws-elastic-beanstalk-2e5e481155de)

```yaml
container_commands:
  AddGlobalWSGIGroupAccess:
    command: "if ! grep -q 'WSGIApplicationGroup %{GLOBAL}' ../wsgi.conf ; then echo 'WSGIApplicationGroup %{GLOBAL}' >> ../wsgi.conf; fi;"
```

When starting the app, you may see the below error because NLTK has not been installed and the corresponding file have not been downloaded.

```bash
LookupError:
**********************************************************************
  Resource \x1b[93mpunkt\x1b[0m not found.
  Please use the NLTK Downloader to obtain the resource:

  \x1b[31m>>> import nltk
  >>> nltk.download('punkt')
  \x1b[0m
  Searched in:
    - '/home/wsgi/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/local/share/nltk_data'
    - '/usr/lib/nltk_data'
    - '/usr/local/lib/nltk_data'
    - '/opt/python/run/venv/nltk_data'
    - '/opt/python/run/venv/lib/nltk_data'
    - ''
**********************************************************************
```

To install NLTK, we have to SSH to the EC2 instance created in beanstalk.

Connect to the EC2 instance using ssh

```bash
ssh -i "<Your pem file>" ec2-user@<Your EC2 instance>
```

If it is successful, you can see the below message.

```bash
 _____ _           _   _      ____                       _        _ _
| ____| | __ _ ___| |_(_) ___| __ )  ___  __ _ _ __  ___| |_ __ _| | | __
|  _| | |/ _` / __| __| |/ __|  _ \ / _ \/ _` | '_ \/ __| __/ _` | | |/ /
| |___| | (_| \__ \ |_| | (__| |_) |  __/ (_| | | | \__ \ || (_| | |   <
|_____|_|\__,_|___/\__|_|\___|____/ \___|\__,_|_| |_|___/\__\__,_|_|_|\_\
                                       Amazon Linux AMI

This EC2 instance is managed by AWS Elastic Beanstalk. Changes made via SSH
WILL BE LOST if the instance is replaced by auto-scaling. For more information
on customizing your Elastic Beanstalk environment, see our documentation here:
http://docs.aws.amazon.com/elasticbeanstalk/latest/dg/customize-containers-ec2.html
```

Install NLTK.

```bash
sudo pip install -U nltk
```

Download the NLTK data to the directory **/usr/local/share/nltk_data**.

```bash
sudo python -m nltk.downloader -d /usr/local/share/nltk_data all
```

After downloading the NLTK data, the app should be able to start and run.

![](/images/dashboard.png)

If it still fails, go to view the logs by navigating to **Logs**, and choose **Request Logs** > **Last 100 lines**


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
