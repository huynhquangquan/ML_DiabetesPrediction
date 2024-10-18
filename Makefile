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

# Review/evaluate model prediction
predict:
ifeq ($(ML),rf)
	python -m src.prediction.prediction_RF
else ifeq ($(ML),logr)
	python -m src.prediction.prediction_LogR
endif

# Run all scripts
run-all: preprocess train evaluate


#--------------------------Clean---------------------------
# Variables for Operating System
UNIX = $(UNIX)
WINDOW = del /Q

# OS=win or OS=unix
# Clear processed data
clean-processed-data:
ifeq ($(OS),win)
	$(WINDOW) data\processed\*
else ifeq ($(OS),unix)
	$(UNIX) data/processed/*
endif

# Clear models results
clean-model-results:
ifeq ($(OS),win)
	$(WINDOW) results\figures\*
	$(WINDOW) results\reports\*
	$(WINDOW) models\*
else ifeq ($(OS),unix)
	$(UNIX) results/figures/*
	$(UNIX) results/reports/*
	$(UNIX) models/*
endif

clean-all: clean-processed-data clean-model-results

# Example: make clean-processed-data OS=win

#-----------------API(application programming interface)------------------
run-api:
	python -m app.API