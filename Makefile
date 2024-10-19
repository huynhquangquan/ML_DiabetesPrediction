#-----------------Download Data------------------
download-raw-data:
	python -m src.dataset.download_data

#-----------------Target to run the model pipeline------------------

# ML=rf or ML=logr

# Preprocess the data
preprocess:
ifeq ($(ML),rf)
	python -m src.preprocessing.data_preprocessing_RF
else ifeq ($(ML),logr)
	python -m src.preprocessing.data_preprocessing_LogR
endif

# Train model
train:
ifeq ($(ML),rf)
	python -m src.models.RandomForest
else ifeq ($(ML),logr)
	python -m src.preprocessing.data_preprocessing_LogR
endif

# Evaluate model
evaluate:
ifeq ($(ML),rf)
	python -m src.evaluate.evaluate_RF
else ifeq ($(ML),logr)
	python -m src.evaluate.evaluate_LogR
endif

# Review evaluated model prediction
predict:
ifeq ($(ML),rf)
	python -m src.prediction.prediction_RF
else ifeq ($(ML),logr)
	python -m src.prediction.prediction_LogR
endif

visualize:
ifeq ($(ML),rf)
	python -m src.visualization.RF_visualization
else ifeq ($(ML),logr)
	python -m src.visualization.LogR_visualization
endif

# Run all scripts
run-all: preprocess train evaluate predict visualize


#--------------------------Clean---------------------------
# Variables for Operating System
UNIX = $(UNIX)
WINDOW = del /Q

# OS=win or OS=unix
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