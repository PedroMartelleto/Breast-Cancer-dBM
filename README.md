# Medical Ultrasound Images for Breast Cancer Classification

----

<div>
<video src="https://github.com/PedroMartelleto/Breast-Cancer-dBM/assets/35240934/8536a58d-6c88-440a-91b6-3b9212577af9" autoplay="true" loop="true"></video>
</div>

## Getting started
The core idea of this project is to revisit the breast cancer classification in order to identify potential so called "digital biomarkers" (dBM). These biomarkers describe parts of the input/features (e.g. a certain regions of the US image) that show a high correlation with target classes. In the field of breast cancer classification we have an obvious BM, namely the cancer itself. Other than that, other parts of the input could pose a high risk for breast cancer or occur as a side effect of breast cancer. Many medical issues show such second order indicators, which are not directly part of the disease itself but connected.

Using breast cancer classification as an example, the task is to set up a complete ML pipeline that could be integrated into an actual application. Especially in the medical field, explainable AI (ExAI) is of high relevance. Therefore, in addition to classifying the images, the task is to highlight and "explain" the models' findings. 

## [M1] Basic ML Pipeline 
- [x] Model selection & Preprocessing
- [x] Model refinement & HP tuning
- [x] Evaluation: Cross-Validation / multiple runs
- [x] Making model accessible via REST-API


## [M2] Explainability Components 
- [x] Selection of matching approaches
- [x] Qualitative evaluation

## [M3] Web based frontend
- [x] Basic Input / Output: Set up a basic website that offers a dialog to upload an image and submits that image to the model. Afterwards models results should be presented on the website.
- [x] Hyperparameter configuration: Some models and exAI approaches require to adapt hyperparameters to show optimal results. Integrate necessary input fields (sliders, text fields etc.) to adapt these HPs.
- [x] Integrate exAI insights: Most exAI methods use a saliency map to highlight ROIs. Integrate these maps into your website

## [M4] Bonus
- [x] Optional: ML applications set up in practical use are designed mostly in a way that users could provide feedback for the presented results. This gives opportunity to additional finetuning after a certain time of collecting new data. To allow a user to give feedback, the web interface should be complemented with an option to edit the saliency map, in order to tell the model if a certain area is not related to the classification or a ROI is missed. 

## Dataset
- https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

## Authors

- Pedro M B Rezende (@PedroMartelleto)
- A special thanks to Christoph Balada (PhD Reseacher @ Deutsches Forschungszentrum für Künstliche Intelligenz) for the project idea, defining milestones, guidance and support.
