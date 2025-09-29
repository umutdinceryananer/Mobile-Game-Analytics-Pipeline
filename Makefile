#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mobile_game_analytics_pipeline
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Rebuild synthetic dataset (clean_data.csv, events.parquet)
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m mobile_game_analytics_pipeline.dataset

## Generate feature and label tables for modeling
.PHONY: features
features:
	$(PYTHON_INTERPRETER) -m mobile_game_analytics_pipeline.features

## Generate analytics tables (funnel, ROI, retention)
.PHONY: analytics
analytics:
	$(PYTHON_INTERPRETER) -m mobile_game_analytics_pipeline.analytics

## Train churn model and persist artefacts
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m mobile_game_analytics_pipeline.modeling.train

## Produce churn predictions using latest model
.PHONY: predict
predict:
	$(PYTHON_INTERPRETER) -m mobile_game_analytics_pipeline.modeling.predict

## Run full pipeline (data -> features -> analytics -> train)
.PHONY: pipeline
pipeline: data features analytics train

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
