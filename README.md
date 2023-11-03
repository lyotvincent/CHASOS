CHASOS: A novel approach for processing inconsistencies in chromatin loop predictions
===
A novel method called CHASOS (CHromatin loop prediction with Anchor Score and OCR Score) to achieve accuracy **chromatin loop predictions**. CHASOS incorporates multi-receptive field large kernel convolutional modules and scale modules to extract features used for **dealing with inconsistencies among multiple feature data**. Using the extracted features, CHASOS constructs a gradient boosting tree model to accomplish the chromatin loop predictions **without introducing extra sequencing**.

# Menu
* /experiments - the source code of comparison methods
* /source/pretrained_model - source code of anchor score model
* /source/fine_tuned_ocr_model - source code of OCR score model
* /data/ChIA-PET/CTCF/raw_data - raw data of detect chromatin loops used for training and testing
* /data/ChIA-PET2_result - the result of chromatin loop detection by ChIA-PET2
* /data/ChIP-seq - ChIP-seq data of CTCF used in chromatin loops filtering
* /data/DNase - DNase data used in chromatin open region prediction in OCR score model
* /data/FIMO_result - the result of motif scanning by FIMO
* /data/negative_loop - the negative chromatin loops used in training and testing
* /data/phastcons - sequence conservation score used in chromatin loop predictions
* /data/positive_loop - the positive chromatin loops used in training and testing
* /data/pretrained_data - the data used in training anchor score model
* /ref_block - some DL block tested in search space of model construction

# Anchor score model
* main training code: /source/pretrained_model/trainer.py
* model code: /source/pretrained_model/models.py - AnchorModel_v20230516_v1
* designed module in model: /source/pretrained_model/custom_layer.py
* training data preparation code: /source/pretrained_model/pretrained_data_loader.py

# OCR score model
* main training code: /source/fine_tuned_ocr_model/ocr_trainer.py
* model code: /source/fine_tuned_ocr_model/ocr_models.py - OCRModel_v20230524_v1
* designed module in model: /source/pretrained_model/custom_layer.py
* training data preparation code: /source/fine_tuned_ocr_model/ocr_data_loader.py

# Loop prediction model
* main training code: /source/loop_model/loop_ml_model_trainer.py
* training data preparation code: /source/loop_model/preprocess_data.py
* drawing K562 loop prediction example code (Figure 5 in paper): /source/loop_model/raw_predictor.py

# Training environment
The anchor score model is trained efficiently on a NVIDIA GeForce RTX 3060 12G GPU
The models are implemented in a Python 3.8 environment, utilizing PyTorch 1.12 as the backend framework.

