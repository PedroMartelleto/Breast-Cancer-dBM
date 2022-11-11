# Medical Ultrasound Images for Breast Cancer Classification
 <details close><summary>Project topic description</summary>

[ChBa_Medical_Ultrasound_Images_for_Breast_Cancer_Classification.pdf](:/ChBa_Medical_Ultrasound_Images_for_Breast_Cancer_Classification.pdf)</details>
----

## Getting started
Core idea of this project is to revisit the breast cancer classification in order to identify potential so called "digital biomarkers" (dBM). These biomarkers describe parts of the input/features (e.g. a certain regions of the US image) that show a high correlation with target classes. In the field of breast cancer classification we have an obvious BM, namely the cancer itself. Other than that, other parts of the input could pose a high risk for breast cancer or occur as a side effect of breast cancer. Many medical issues show such second order indicators. Indicators, that are not part of the disease itself but connected.

Using breast cancer classification as example, your task is to set up a complete ML pipeline that could be integrated into an actual application. Especially in the medical field, explainable AI (ExAI) is of high relevance. Therefore, in addition to classifying the images, your task is to highlight and "explain" your models findings. 

- [ ] DFKI Account
	- DataPrivacySheet.pdf
	- CommitmentAccountHolder.pdf
	- ID
- [ ] Cluster Account
- [ ] Weekly or biweekly meeting?


## [M1] Basic ML Pipeline 

The task of this first milestone is to assemble your general training framework including a modular data loader, the possibility for hyperparameter (HP) tuning, and proper reporting, evaluation, and logging. This phase is also used to familiarise yourself with the compute cluster and train your first baseline models.

- [ ] Model selection & Preprocessing
- [ ] Model refinement & HP tuning
- [ ] Evaluation: Cross-Validation / multiple runs
- [ ] Making model accessible via REST-API


## [M2] Explainability Components 
The task of this second milestone is to familiarise yourself with Captum, a framework for inspecting predictions your trained models. Furthermore, it is your task to identify ExAI approaches that fit the use case of breast cancer classification. 

- [ ] Selection of matching approaches
- [ ] Qualitative evaluation

## [M3] Web based frontend
Nowadays, nearly everything runs based on small web services. In particular RESTfull services proofed as reliable method to provide a service in the context of an ML application. Therefore, task of the third milestone is to set up a small web-based frontend. This frontend should allow a user to upload a new input image, configure HPs and visualise the models output including exAI insights. 
*Note: no over-engineering needed - keep it simple but useful. Some basic interfaces built with plotly is absolutely fine*

- [ ] Basic Input / Output: Set up a basic website that offers a dialog to upload an image and submits that image to the model. Afterwards models results should be presented on the website.
- [ ] Hyperparameter configuration: Some models and exAI approaches require to adapt hyperparameters to show optimal results. Integrate necessary input fields (sliders, text fields etc.) to adapt these HPs.
- [ ] Integrate exAI insights: Most exAI methods use a saliency map to highlight ROIs. Integrate these maps into your website

## [M4] Bonus
The project offers tons of potential improvements across all milestones.
- [ ] Optional: Additional Data for unsupervised pretraining. Using a more sophisticated training scheme additional gain in performance could be achieved. One option is to use an un- or self-supervised pretraining. Approaches like BYOL, SimCLR or SWAV could offer necessary tools. 
- [ ] Optional: Depending on experimental findings, additional analysis on exAI saliency maps could be performed. In particular spatial relations of saliency hot spots could be of interest. 
- [ ] Optional: ML applications set up in practical use are designed mostly in a way that users could provide feedback for the presented results. This gives opportunity to additional finetuning after a certain time of collecting new data. To allow a user to give feedback, the web interface should be complemented with an option to edit the saliency map, in order to tell the model if a certain area is not related to the classification or a ROI is missed. 
- [ ] Data
	- https://www.kaggle.com/datasets/vuppalaadithyasairam/ultrasound-breast-images-for-breast-cancer
	- https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
	- https://www.kaggle.com/datasets/ignaciorlando/ussimandsegm
