#-----------------Download Data and Create External Data------------------
download-raw-data:
	python -m src.dataset.download_data

#-----------------Target to run the model pipeline------------------
# Preprocess the data
preprocess:
	python -m src.preprocessing.data_preprocessing

# Train model
train.%: # Ex: make train.RandomForest (correct model name)
	python -m src.models.$*

# Evaluate model
evaluate:
	python -m src.evaluate.evaluate

# Review evaluated model prediction
predict:
	python -m src.prediction.prediction

# Visualize evaluated model
visualize:
	python -m src.visualization.visualization

# Run combined scripts
model ?= selected_model # Ex: make model=RandomForest run-all
create-data: download-raw-data preprocess
total-evaluation: train.$(model) evaluate visualize predict
total-evaluation-plus:  preprocess train.$(model) evaluate visualize predict

#--------------------------Clean---------------------------
# Variables for Operating System
UNIX = rm -rf
WINDOW = del /Q

# OS=win or OS=unix
clear-external-data:
ifeq ($(OS),win)
	$(WINDOW) data\external\*
else ifeq ($(OS),unix)
	$(UNIX) data/external/*
endif

# Clear processed data
clear-processed-data:
ifeq ($(OS),win)
	$(WINDOW) data\processed\*
else ifeq ($(OS),unix)
	$(UNIX) data/processed/*
endif

# Clear raw data
clear-raw-data:
ifeq ($(OS),win)
	$(WINDOW) data\raw\*
else ifeq ($(OS),unix)
	$(UNIX) data/raw/*
endif

# Clear models results
clear-models-and-results:
ifeq ($(OS),win)
	$(WINDOW) results\figures\*
	$(WINDOW) results\reports\*
	$(WINDOW) models\*
else ifeq ($(OS),unix)
	$(UNIX) results/figures/*
	$(UNIX) results/reports/*
	$(UNIX) models/*
endif

clear-all: clear-raw-data clear-processed-data clear-models-and-results

# Example: make clean-processed-data OS=win

#-----------------API(application programming interface)------------------
run-api:
	python -m app.API

run-uvicorn:
	uvicorn app.API:API --reload --factory
