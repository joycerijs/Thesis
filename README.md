**Purpose**

This repository was created to contain the source code of the MSc Technical Medicine (joint degree: TU Delft, Erasmus MC, LUMC) Thesis of Joyce Rijs. The source code
contains methods for automated processing of EHR correspondence for the development of a Machine Learning (ML) model for Intellectual Disability detection.
The automated processing consists of text extraction, de-identification and two types of feature extraction methods: bag-of-words and clinical concept extraction.

This project was performed by Joyce Rijs from September 2023 - March 2024. The project was initiated by Laura de Graaff (Internist-Endocrinologist at Erasmus MC)
and Jifke Veenland (Researcher and Education Innovator at Erasmus MC).

**Folders and files**

- De-identification folder
    - Datasets folder: This folder contains csv files of the datasets with sensitive keywords that were masked during de-identification.
    - De-identification.py: This script was used to extract and de-identify text of the following files: PDF, doc, docx, odt, txt, png, tiff, jpg and jpeg.
    - filter.yaml: This yaml file can be used to initialize the filter for de-identification.
    - Preprocess_names.py: This script was used to preprocess Novicare client names for addition to the keywords lists in the datasets folder. NOTE: the client names were not included in the dataset folder of this repository for privacy reasons.
    - Processor.py: This script contains the KeywordProcessor class for processing keywords and was fully adopted from L. van der Meulen.
- BOW_pipeline.py: With this script, all bag-of-words processing steps were performed and a ML model was trained and evaluated. 
                    Furthermore, client age, gender, amount of files and amount of words were statistically compared.
- CCE_pipeline.py: With this script, the UMLS codes obtained with the MedSpacy toolkit were processed and a ML model was trained and evaluated. NOTE: The English dependencies and language context rules of MedSpacy were replaced by Dutch versions by T. Seinen (1). 
                    This Dutch MedSpacy toolkit was not published yet and therefore not included in this repository.
- Statictics_and_ML.py: This script contains the functions developed for calculating statistics and training and evaluating ML models.

(1) Seinen TM, Kors JA, van Mulligen EM, Fridgeirsson E, Rijnbeek PR. The added value of text from Dutch general practitioner notes in predictive modeling. J Am Med Inform Assoc. 2023 Nov 17;30(12):1973–84.

_If something is not clear, please send an e-mail to joyce.rijs@hotmail.com_
