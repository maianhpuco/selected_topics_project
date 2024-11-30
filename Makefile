# Define variables for training parameters
include .env
export 


DATASET_PATH := "/project/hnguyen2/mvu9/datasets/chexpert"
DATA_FOLDER :="/project/hnguyen2/mvu9/datasets/chexpert/CheXpert-v1.0-small" 
# RAWDATA_PATH := ../data/raw
RAWDATA_PATH := "/project/hnguyen2/mvu9/datasets/chexpert/raw/" 
MODEL_NAME := resnet50
BATCH_SIZE := 128
EPOCHS := 50
LEARNING_RATE := 1e-4
WEIGHT_DECAY := 1e-5

# Kaggle dataset and paths
KAGGLE_DATASET := willarevalo/chexpert-v10-small
KAGGLE_JSON := ./kaggle.json
KAGGLE_INSTALL_CMD := pip install kaggle


set_up: $(KAGGLE_JSON)
	$(KAGGLE_INSTALL_CMD)
	chmod 600 $(KAGGLE_JSON) 		  


# Default target to train the model
train: $(DATASET_PATH)
	python train.py --dataset_path $(DATASET_PATH) --model_name $(MODEL_NAME) --batch_size $(BATCH_SIZE) --epochs $(EPOCHS) --lr $(LEARNING_RATE) --weight_decay $(WEIGHT_DECAY)
 
# Download data from Kaggle
download_data:
	kaggle datasets download -d $(KAGGLE_DATASET) -p $(DATASET_PATH)
	unzip $(DATASET_PATH)/chexpert-v10-small.zip -d $(RAWDATA_PATH)

# Dependency: Check if kaggle.json exists
$(KAGGLE_JSON):
	@echo "Please provide kaggle.json in the current directory for authentication."

# Clean logs, checkpoints, and data
clean:
	rm -rf ./logs ./checkpoints ./data

# Example target to test the model (modify with your test script details)
test:
	python test_model.py --model_path ./checkpoints/model.pth --image_path ./sample_image.jpg
 