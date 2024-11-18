import tensorflow as tf
import optuna
import time
import logging
import numpy as np
from pathlib import Path
import pandas as pd
import os
from src.low_cost_bnn.workflows.train_tensorflow_ncp import train_tensorflow_ncp_epoch, train_tensorflow_ncp, launch_tensorflow_pipeline_ncp
from src.low_cost_bnn.utils.pipeline_tools import setup_logging, print_settings, preprocess_data
from src.low_cost_bnn.utils.helpers_tensorflow import default_dtype, create_data_loader, create_scheduled_adam_optimizer, create_regressor_model, create_regressor_loss_function, wrap_regressor_model, save_model

#objective function
def objective(trial):


	#define hyperparameters (HPs) to be hypertuned using trial.suggest method

	ood_width = trial.suggest_float('ood_sampling_width',0.01,0.2)
	epi_weight = [trial.suggest_float('epistemic_weight',1e-4,1.0,log=True)]
	alea_weight = [trial.suggest_float('aleatoric_weight',1e-4,1.0,log=True)]
	l1_reg = trial.suggest_float('l1_regularization',0.0,0.5)
	reg_weight = trial.suggest_float('regularization_weight',0.01,10,log=True)
	n_layers = trial.suggest_int('num_layers',3,7)
	num_neurons = trial.suggest_int('num_neurons',320,1024,log=True)
	num_layers_list = [num_neurons] * n_layers 
	#epi_prior = [trial.suggest_loguniform('epistemic_priors',1e-5,1e-2)]
	#alea_prior = [trial.suggest_loguniform('aleatoric_priors',1e-5,1e-2)]

	data = pd.read_hdf('/pool001/vgalvan/combined_training_data_expanded_RMIN/rho_07/combined_training_data_1_Mil_20per_wiggle_PRD_LMODE_rho07_shuffled_0.h5')
	data = data.loc[(data['Q_PRIME_LOC'] < 50.0)&(data['Q_LOC']<3.0)&(data['Q_E_Target']<20.0)]
	#data = data.loc[data['Q_E_Target'] < 20.0]
	input_var = ['RMIN_LOC','RMAJ_LOC','KAPPA_LOC','S_KAPPA_LOC','DELTA_LOC','DRMAJDX_LOC','Q_LOC','Q_PRIME_LOC','BETAE', 
					'XNUE','ZEFF','TAUS_2','AS_2','AS_3','RLTS_1','RLTS_2','RLNS_2','P_PRIME_LOC']
	output_var = ['Q_E_Target']


	#Uncomment this section if the metrics and model of each trial needs to be saved.
	#Needs additional development. Recommended to be implemented in conjunction with individual job array id. 
	'''
	mfile = 'metrics_tf_bnn_'+str(count)+'.h5' #metrics file
	nfile = 'model_tf_bnn_'+str(count)+'.keras' #model file

	'''

	#instantiate model and metrics dictionary calling launch_tensor_flow_pipeline_ncp function directly
	#use the same trial.suggest HPs from above as arguments

	trained_model, metrics_dict = launch_tensorflow_pipeline_ncp(
		data=data,
		input_vars=input_var,
		output_vars=output_var,
		validation_fraction=0.2,
		test_fraction=0.1,
		data_split_file=None,
		max_epoch=100,
		batch_size=9999,
		early_stopping=100,
		minimum_performance=None,
		shuffle_seed=None,
		sample_seed=None,
		generalized_widths=num_layers_list,
		specialized_depths=None,
		specialized_widths=None,
		l1_regularization=l1_reg,
		l2_regularization=0.9,
		relative_regularization=0.1,
		ood_sampling_width=ood_width,
		epistemic_priors=[1e-3],
		aleatoric_priors=[1e-3],
		distance_loss='fisher_rao',
		likelihood_weights=[0.01],
		epistemic_weights=epi_weight,
		aleatoric_weights=alea_weight,
		regularization_weights=reg_weight,
		learning_rate=1.0e-6,
		decay_rate=0.95,
		decay_epoch=20,
		checkpoint_freq=0,
		checkpoint_dir=None,
		training_device='gpu',
		verbosity=0
		)

	valid_r20 = metrics_dict['valid_r20'].iloc[-1] #grabbing the r^2 value at the end of each batch


	#Uncomment this section if the metrics and model of each trial needs to be saved.
	'''
	metrics_dict.to_hdf(mfile, key='/data')

	save_model(trained_model, nfile)
	'''
	
	#return the objective metric (in this case it is val r^2 from metrics dict)
	return valid_r20



#Create a storage object for trial metrics (this needs to be done this way to deal with NFS file system on engagaing)
database_path = '/pool001/vgalvan/low-cost-bnn/training_runs/low_flux_Q_LOC_filter_model_hp_tuning_7_params/7_hp_parallel_study.db'
lock_obj = optuna.storages.JournalFileOpenLock(database_path)
storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(database_path,lock_obj=lock_obj))

#Create a study object
#If running many trials in parallel set 'n_trials=1' and use a job array in batch job script to set desired amount of trials 
study = optuna.create_study(storage=storage,study_name='parallel_optuna_test_1Mil_low_flux_database',direction='maximize',load_if_exists=True)
study.optimize(objective,n_trials=1)

trial = study.best_trial

#optional output
print('r2: {}'.format(trial.value))
print('Best hyperparameters: {}'.format(trial.params))
