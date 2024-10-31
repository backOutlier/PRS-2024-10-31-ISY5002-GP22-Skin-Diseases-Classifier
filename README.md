# PRS-2024-10-31-ISY5002-GP22-Skin-Diseases-Classifier

![img](Miscellaneous/sd_header.jpg)

------

## SECTION 1 : EXECUTIVE SUMMARY / PAPER ABSTRACT

------
`    `This project develops an AI-powered diagnostic system for skin disease classification. By using advanced deep learning models in image classification area—ResNet50, DenseNet121, and VGG16, the system achieves a diagnostic accuracy range of 55-70% across 23 skin disease categories. And by further applying a hard-voting ensemble method, it combines model strengths, enhancing the precision and robustness of skin disease identification, notably improving classification performance for common conditions like acne, nail fungus, and melanoma. 
  
  `    `Additionally, the system is deployed as an interactive web application, allowing healthcare providers and patients to upload images and receive real-time diagnostic feedback in a streamlined manner. The front-end is built using Vue.js for its flexibility and responsiveness, combined with Vite for fast-build development, which enhances the user experience. The interface uses Axios to handle asynchronous HTTP requests, enabling smooth data transmission between the front-end and the Django-based backend. To improve accessibility and foster trust, the system incorporates interpretable results, supporting clinicians in complex cases and providing patients with a preliminary self-assessment tool.   
  
`    `Future work includes broadening the dataset to ensure representation across skin tones, age groups, and rare conditions, as well as integrating multimodal data—such as patient history and clinical test results—for a more comprehensive diagnostic view. Incorporating a real-time feedback mechanism will also allow for continuous model retraining, improving diagnostic accuracy across diverse clinical applications. This AI-based diagnostic system sets a new standard in accessible, accurate dermatology tools, benefiting both medical professionals and patients worldwide.


## SECTION 2 : CREDITS / PROJECT CONTRIBUTION



| Official Full Name | Student ID (MTech Applicable) | Work Items (Who Did What)                                    | Email (Optional)                                      |
| ------------------ | ----------------------------- | ------------------------------------------------------------ | ----------------------------------------------------- |
| Mohan Liu          | A0297443U                     | 1. The process of cleaning data, extracting features, training models (such as ResNet50), and evaluating models. <br>2. Develop integrated models and produce hard voting algorithms to integrate the most accurate algorithms, as well as prepare implementation documents.<br> 3. Contribute to the preparation(write and revise) of reports and the design of their layout. <br>4. Delegate tasks and monitor progress. | [e1351581@u.nus.edu](mailto:e1351581@u.nus.edu)       |
| Yuhao Zhou         | A1234567B                     | 1. Literature review  <br>2. Training MobileNet model <br>3. Server transmission testing<br> 4. Part of report writing <br>5.Demo recording<br>6.Web application front-end development | [zhouyuhao24@u.nus.edu](mailto:zhouyuhao24@u.nus.edu) |
| LiXin Zhang        | A0279544N                     | 1. Participate in system design discussions and draw system architecture diagrams <br>2. Participate in model design and write reports on model training part <br>3. Participate in report integration <br>4. Produce PowerPoint and video for system design section | [E1351682@u.nus.edu](mailto:E1351682@u.nus.edu)       |
| Zhiyuan Zhang      | A0297736J                     | 1. project reproduction, project Intro, data collection <br>2. model training <br>3.related report writing | [e1351874@u.nus.edu](mailto:e1351874@u.nus.edu)       |
| Wenyu Zhong         | A0294636R                     | 1.mainly responsible for web application backend development<br>2. nvolved in integration of the model<br>3.participating in report writing and ppt video creation| [e1348774@u.nus.edu](mailto:e1348774@u.nus.edu)       |

------

------

## SECTION 3 : USER GUIDE
`Refer to appendix <Installation & User Guide> in project report at Github Folder: ProjectReport`

Make sure all developer tools have been installed:

- npm
- Python3
- pip

### [ 1 ] To run the back-end server:

```
$ cd SystemCode/backend
$ pip install -r requirements.txt
$ cd myproject
$ python manage.py makemigrations api
$ python manage.py makemigrations
$ python manage.py migrate
$ python manage.py runserver
```

### [ 2 ] To run the front-end server:

```
$ cd SystemCode/frontend
$ npm install
$ npm run dev
```

> **Go to URL using web browser** [http://127.0.0.1:4000](http://127.0.0.1:4000/)

