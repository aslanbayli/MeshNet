TYPE ?= non_mesh
GPU ?= 0

features:
	python src/features/build_features.py

# put 'mesh' for the 'type' to generate a mesh dataset
dataset:
	python src/data/make_dataset.py ${TYPE}

train:
	CUDA_VISIBLE_DEVICES=${GPU}, python src/models/train_model.py 

train-nohup:
	CUDA_VISIBLE_DEVICES=${GPU}, nohup python src/models/train_model.py > src/experiments/nohup5.out &

pred:
	CUDA_VISIBLE_DEVICES=${GPU}, python src/models/predict_model.py ${e} ${lr} ${bs} ${hs}

vis:
	python src/visualization/visualize.py ${op} ${line}