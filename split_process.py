import tensorflow as tf
import tensorflow_federated as tff
#import tensorflow_federated.simulation.FileCheckpointManager as ckpt
import tensorflow.keras as tk
import collections
import os
import pandas as pd
import numpy as np
import datetime
from keras.models import load_model
from sklearn import metrics
from itertools import zip_longest
#from chexpert_parser import load_dataset, feature_description
from load_dataset import make_dataset


class LabelAUC(tf.keras.metrics.AUC):
	'''
	Custom AUC metric of a specific pathology
	:param label_id: id of a specific pathology
	'''
    def __init__(self, label_id, name="label_auc", **kwargs):
        super(LabelAUC, self).__init__(name=name, **kwargs)
        self.label_id = label_id
     
    def update_state(self, y_true, y_pred, **kwargs):
        return super(LabelAUC, self).update_state(y_true[:, self.label_id], y_pred[:, self.label_id], **kwargs)
     
    def result(self):
        return super(LabelAUC, self).result()

class MeanAUC(LabelAUC): 
	''' Custom AUC metric that computes the Mean Auc of the interest pathology '''
    def __init__(self, label_id, name="label_mean_auc", **kwargs):
        super(MeanAUC, self).__init__(label_id=label_id, name=name, **kwargs)
        self.aucs = [LabelAUC(label_id=label_id[0]), LabelAUC(label_id=label_id[1]), LabelAUC(label_id=label_id[2]), LabelAUC(label_id=label_id[3]), LabelAUC(label_id=label_id[4])]

    def update_state(self, y_true, y_pred, **kwargs):
        for auc in self.aucs:
            auc.update_state(y_true, y_pred)
    
    def result(self):
        return tf.reduce_mean([auc.result().numpy() for auc in self.aucs])

    def reset_states(self):
        return super(LabelAUC, self).reset_states()


class SplitProcess():
	def __init__(self, model_name, model_architecture, input_shape=(224,224,3), output_shape=(14,), checkpoint_folder='./models/'):
		''' - Represents a Split Process - '''
		self.model_name = model_name
		self.model_architecture = model_architecture
		self.number_of_clients = None # Will be populated according to the dataset dictionary
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.checkpoint_folder = None
		self.current_round=1

	def setup(self, dataset_paths, val_dataset_paths, output_folder):
		'''
		Setup the split process from a dictionary containing the client configuration. The top-level keys are the client names, the corresponding keys are dictionaries with a set of parameters: 'path' - the path of the .tfrecord to load, 'take_only': How many samples to take from the dataset (truncation - can be None)
		:param dataset_paths: a dictionary of dictionaries, example {'client_1':{'path': str, 'take_only': None}}
		:param val_dataset_paths: a dictionary of dictionaries containing validation datasets
		:param output_folder: string 
		'''
		self.number_of_clients = len(dataset_paths)
		self.client_list = sorted(dataset_paths.keys())

		print("Creating split dataset for {} clients".format(self.number_of_clients))
		self.train_datasets = {client: make_dataset(dataset_paths[client]['path']).batch(64, drop_remainder=False).prefetch(1) for client in self.client_list}
		#self.train_datasets = {client: load_dataset(dataset_paths[client]['path'], debug=True, take=dataset_paths[client]['take_only'] if 'take_only' in dataset_paths[client] else None) for client in self.client_list}
		#self.val_datasets = {client: load_dataset(val_dataset_paths[client]['path'], debug=True, take=val_dataset_paths[client]['take_only'] if 'take_only' in val_dataset_paths[client] else None) for client in self.client_list}
		self.val_datasets = make_dataset(val_dataset_paths['path']).batch(64, drop_remainder=False).prefetch(1)

		if not os.path.exists(output_folder):
		    os.makedirs(output_folder)
		    #os.makedirs(output_folder+'/checkpoint')
		    os.makedirs(output_folder+'/tensorboard_log/train')
		    os.makedirs(output_folder+'/tensorboard_log/valid')
		self.path = output_folder
		#self.checkpoint_folder = output_folder+'/checkpoint'
		self.tb_path_train = output_folder+'/tensorboard_log/train'
		self.tb_path_valid = output_folder+'/tensorboard_log/valid'
		self.tb_writer_t = tf.summary.create_file_writer(self.tb_path_train)
		self.tb_writer_v = tf.summary.create_file_writer(self.tb_path_valid)
		self.models_dict = {}
		for client in self.client_list:
			print("Builiding model of {}...".format(client))
			self.models_dict[client] = self.build_model()

		self.metrics = {
			'auc_train' : tf.keras.metrics.AUC(name='auc_train'),
			'mean_auc_train' : MeanAUC(label_id=[2,5,6,8,10], name='mean_auc_train'),
			'auc_train_card' : LabelAUC(label_id=2, name='auc_train_card'),
			'auc_train_edema' : LabelAUC(label_id=5, name='auc_train_edema'),
			'auc_train_cons' : LabelAUC(label_id=6, name='auc_train_cons'),
			'auc_train_atel' : LabelAUC(label_id=8, name='auc_train_atel'),
			'auc_train_peff' : LabelAUC(label_id=10, name='auc_train_peff'),

			'auc_valid' : tf.keras.metrics.AUC(name='auc_valid'),
			'mean_auc_valid' : MeanAUC(label_id=[2,5,6,8,10], name='mean_auc_valid'),
			'auc_valid_card' : LabelAUC(label_id=2, name='auc_valid_card'),
			'auc_valid_edema' : LabelAUC(label_id=5, name='auc_valid_edema'),
			'auc_valid_cons' : LabelAUC(label_id=6, name='auc_valid_cons'),
			'auc_valid_atel' : LabelAUC(label_id=8, name='auc_valid_atel'),
			'auc_valid_peff' : LabelAUC(label_id=10, name='auc_valid_peff'),
		}

		self.metrics_name = [met for met in self.metrics.keys()]
		self.loss_fn = tf.keras.losses.BinaryCrossentropy()
		self.optimizer = tf.keras.optimizers.SGD(1e-3)
	
	
	def build_model(self):
		'''
		Builds a keras model based on the model architecture provided in the init function
		:return: A Keras model
		'''
		base_model = self.model_architecture(input_shape=self.input_shape, weights='imagenet', include_top=False)
		x = base_model.output
		x = tk.layers.GlobalAveragePooling2D()(x)
		predictions = tk.layers.Dense(14, activation='sigmoid')(x)
		return tk.Model(inputs=base_model.inputs, outputs=predictions)


	def compute_metrics(self, y_true, y_pred, client, run):
		'''
		updates metrics based on:
		:param y_true: vector of true output
		:param y_pred: vector of predicted output
		:param client: string, states clients who is runinng the training (or validation)
		:param run: string, states if is training or validation run
		'''
		if run == 'training':
			self.metrics['auc_train'].update_state(y_true, y_pred)
			self.metrics['mean_auc_train'].update_state(y_true, y_pred)
			self.metrics['auc_train_card'].update_state(y_true, y_pred)
			self.metrics['auc_train_edema'].update_state(y_true, y_pred)
			self.metrics['auc_train_cons'].update_state(y_true, y_pred)
			self.metrics['auc_train_atel'].update_state(y_true, y_pred)
			self.metrics['auc_train_peff'].update_state(y_true, y_pred)
		if run == 'validation':
			self.metrics['auc_valid'].update_state(y_true, y_pred)
			self.metrics['mean_auc_valid'].update_state(y_true, y_pred)
			self.metrics['auc_valid_card'].update_state(y_true, y_pred)
			self.metrics['auc_valid_edema'].update_state(y_true, y_pred)
			self.metrics['auc_valid_cons'].update_state(y_true, y_pred)
			self.metrics['auc_valid_atel'].update_state(y_true, y_pred)
			self.metrics['auc_valid_peff'].update_state(y_true, y_pred)


	@tf.function
	def train_step(self, x, y, client):
		'''
		Perform the train step
		:param x: vector of input image
		:param y: vector of true output
		:param client: string, states clients who is runinng the training
		'''
		with tf.GradientTape(persistent=True) as tape:
			output = self.models_dict[client](x, training=True)												
			loss_value = self.loss_fn(y, output)
		grads = tape.gradient(loss_value, self.models_dict[client].trainable_weights)
		self.optimizer.apply_gradients(zip(grads, self.models_dict[client].trainable_weights))
		del tape
		self.compute_metrics(y, output, client, run='training')
		return loss_value

	@tf.function
	def validation_step(self, x, y, client):
		'''
		Perform the validation step
		:param x: vector of input image
		:param y: vector of true output
		:param client: string, states clients who is runinng the validation
		'''
		output = self.models_dict[client](x, training=False)
		loss = self.loss_fn(y, output)
		self.compute_metrics(y, output, client, run='validation')
		return loss

	def save_server_weights(self, client, sl_bottom, sl_top=None):
		'''
		Save server weights in order to broadcast them to other clients.
		:param client: model who is saving server weights
		:param sl_bottom: int, split layer bottom
		:param sl_top: int, split layer top for u-shaped training (not implemented)
		'''
		if sl_top is None:
			weights = []
			for i in range(sl_bottom, len(self.models_dict[client].layers)):
				weights.append(self.models_dict[client].layers[i].get_weights())
		else:
			print("NOT IMPLEMENTED") 
		return weights

	def update_server_weights(self, weights, sl, client_updated="Dummy"):
		'''
		Updates server layers of clients that received weights from client who trained his model.
		:param weights: vector of server weights
		:param sl: int, split layer.
		:param client_updated: client who is broadcasting weights. this client will not update his weights
		'''
		for client, model in self.models_dict.items():
			if client is not client_updated:
				for i in range(len(weights)):
					self.models_dict[client].layers[i + sl].set_weights(weights[i])

	def average_weights(self, sl_bottom):
		'''
		ONLY for medium grain. Average weights from all clients
		:param weights: vector of server weights
		:param sl_bottom: split layer bottm
		'''
		weights = []
		layers = tuple([ l for l in m.layers[60:] if l.trainable] for m in self.models_dict.values())
		for clients_layers in zip(*layers):
			mean_value = np.mean([l.get_weights() for l in clients_layers], axis=0)
			weights.append(mean_value)
		return weights

	def log_epoch(self, log, epoch, run):
		''' Log epoch to a dataframe '''
		stacked = np.stack([met.result().numpy() for met in self.metrics.values()], axis=0)
		step_log = pd.DataFrame(np.array([stacked]), columns=self.metrics_name)
		step_log.insert(0, 'Epoch', epoch)
		log = log.append(step_log, ignore_index=True)
		return log

	def callback_earlyStopping(self, MetricList, min_delta=0.1, patience=20, mode='min'):
		''' 
		callback earlystopping for training 
		:param MetricList: list of the past metrics to check:
		'''

		#No early stopping for the first patience epochs 
		if len(MetricList) <= patience:
			return False
		min_delta = abs(min_delta)
		if mode == 'min':
			min_delta *= -1
		else:
			min_delta *= 1
		#last patience epochs 
		last_patience_epochs = [x + min_delta for x in MetricList[::-1][1:patience + 1]]
		current_metric = MetricList[::-1][0]
		if mode == 'min':
			if current_metric >= max(last_patience_epochs):
				print(f'Metric did not decrease for the last {patience} epochs.')
				return True
			else:
				return False
		else:
			if current_metric <= min(last_patience_epochs):
				print(f'Metric did not increase for the last {patience} epochs.')
				return True
			else:
				return False

	def save_log(self, log):
	''' Save lof to a csv_file '''
		file=self.path+'/log.csv'
		with open(file, mode='w') as f:
			log.to_csv(f, index=False)


# *************************************************************************************************************************** #			

	def iterative_training(self, split_layer, epochs):
		logger = pd.DataFrame()
		metrics_seq = []
		for e in range(epochs):
			print("Start of epoch %d" %(e)) 
			step=0
			# Training
			for row0, row1, row2, row3, row4 in zip_longest(self.train_datasets['client_0'], self.train_datasets['client_1'], self.train_datasets['client_2'], self.train_datasets['client_3'], self.train_datasets['client_4']):
				step+=1
				if row0:
					train_loss = self.train_step(row0[0], row0[1], 'client_0')
					server_weights = self.save_server_weights('client_0', sl_bottom=split_layer)
					self.update_server_weights(server_weights, split_layer, 'client_0')

				if row1:
					train_loss = self.train_step(row1[0], row1[1], 'client_1')
					server_weights = self.save_server_weights('client_1', sl_bottom=split_layer)
					self.update_server_weights(server_weights, split_layer, 'client_1')           

				if row2:
					train_loss = self.train_step(row2[0], row2[1], 'client_2')
					server_weights = self.save_server_weights('client_2', sl_bottom=split_layer)
					self.update_server_weights(server_weights, split_layer, 'client_2')

				if row3:
					train_loss = self.train_step(row3[0], row3[1], 'client_3')
					server_weights = self.save_server_weights('client_3', sl_bottom=split_layer)
					self.update_server_weights(server_weights, split_layer, 'client_3')

				if row4:
					train_loss = self.train_step(row4[0], row4[1], 'client_4')
					server_weights = self.save_server_weights('client_4', sl_bottom=split_layer)
					self.update_server_weights(server_weights, split_layer, 'client_4')

				if step%50==0:
					template = 'TRAINING: Epoch {}, Step {}, AUC train: {}, AUC_Mean: {}, AUC train cardiomegaly: {}'
					print(template.format(e+1, step, self.metrics['auc_train'].result().numpy(), self.metrics['mean_auc_train'].result().numpy(), self.metrics['auc_train_card'].result().numpy()))
			
			# Saving training metrics on tensorboard
			with self.tb_writer_t.as_default():
				for met in dict(list(self.metrics.items())[:7]).items():
					tf.summary.scalar(met[0], met[1].result(), step=e)

			# Validation
			for client in self.client_list:
				#for step, row in enumerate(self.val_datasets[client]):
				for step, row in enumerate(self.val_datasets):
					val_loss = self.validation_step(row[0], row[1], client)
					if step%50==0:
						template = 'VALIDATION: Epoch {}, Step {}, AUC valid: {}, AUC_Mean: {}, AUC_Mean Auc valid card: {}'
						print(template.format(e+1, step, self.metrics['auc_valid'].result().numpy(), self.metrics['mean_auc_valid'].result().numpy(), self.metrics['auc_valid_card'].result().numpy()))

			# Saving validation metrics on tensorboard
			with self.tb_writer_v.as_default():
				for met in dict(list(self.metrics.items())[7:]).items():
					tf.summary.scalar(met[0], met[1].result(), step=e)

			#ckpt.save(outputFolder+"/Model_A_epoch_{}".format(e))

			# Log metrics 
			logger = self.log_epoch(logger, epoch=e+1, run='training')
			print(logger)

			# Check for EarlyStopping
			metrics_seq.append(self.metrics['mean_auc_valid'].result().numpy())
			stopEarly = self.callback_earlyStopping(metrics_seq, min_delta=0.1, patience=4, mode='max')
			if stopEarly:
				print("Callback_EarlyStopping signal received at epoch = %d/%d"%(e+1,epochs))
				print("Terminating training ")
				break

				
			# Reset metrics at each epoch
			for m in self.metrics.values():
				m.reset_states()

		# Save Models and Logs 
		self.models_dict['client_0'].save(self.path+'/client_0.h5')
		self.models_dict['client_1'].save(self.path+'/client_1.h5')
		self.models_dict['client_2'].save(self.path+'/client_2.h5')
		self.models_dict['client_3'].save(self.path+'/client_3.h5')
		self.models_dict['client_4'].save(self.path+'/client_4.h5')
		self.save_log(logger)

	def iterative_training_coarse(self, split_layer, epochs):
		logger = pd.DataFrame()
		metrics_seq = []
		for e in range(epochs):
			print("Start of epoch %d" %(e))
			# Training
			for client in self.client_list:
				for step, row in enumerate(self.train_datasets[client]):
					train_loss = self.train_step(row[0], row[1], client)
					if step%50==0:
						template = 'TRAINING {}: Epoch {}, Step {}, AUC train: {}, AUC_Mean: {}, AUC train cardiomegaly: {}'
						print(template.format(client, e+1, step, self.metrics['auc_train'].result().numpy(), self.metrics['mean_auc_train'].result().numpy(), self.metrics['auc_train_card'].result().numpy()))
				server_weights = self.save_server_weights(client, sl_bottom=split_layer)
				self.update_server_weights(server_weights, split_layer, client)
			
			# Saving training metrics on tensorboard
			with self.tb_writer_t.as_default():
				for met in dict(list(self.metrics.items())[:7]).items():
					tf.summary.scalar(met[0], met[1].result(), step=e)

			# Validation
			for client in self.client_list:
				#for step, row in enumerate(self.val_datasets[client]):
				for step, row in enumerate(self.val_datasets):
					val_loss = self.validation_step(row[0], row[1], client)
					if step%50==0:
						template = 'VALIDATION {}: Epoch {}, Step {}, AUC valid: {}, AUC_Mean: {}, AUC valid Card: {}'
						print(template.format(client, e+1, step, self.metrics['auc_valid'].result().numpy(), self.metrics['mean_auc_valid'].result().numpy(), self.metrics['auc_valid_card'].result().numpy()))

			# Saving validation metrics on tensorboard
			with self.tb_writer_v.as_default():
				for met in dict(list(self.metrics.items())[7:]).items():
					tf.summary.scalar(met[0], met[1].result(), step=e)

			#ckpt.save(outputFolder+"/Model_A_epoch_{}".format(e))

			# Log metrics 
			logger = self.log_epoch(logger, epoch=e+1, run='training')
			print(logger)

			# Check for EarlyStopping
			metrics_seq.append(self.metrics['mean_auc_valid'].result().numpy())
			stopEarly = self.callback_earlyStopping(metrics_seq, min_delta=0.1, patience=4, mode='max')
			if stopEarly:
				print("Callback_EarlyStopping signal received at epoch = %d/%d"%(e+1,epochs))
				print("Terminating training ")
				break
				
			# Reset metrics at each epoch
			for m in self.metrics.values():
				m.reset_states()

		# Save Models and Logs 
		self.models_dict['client_0'].save(self.path+'/client_0.h5')
		self.models_dict['client_1'].save(self.path+'/client_1.h5')
		self.models_dict['client_2'].save(self.path+'/client_2.h5')
		self.models_dict['client_3'].save(self.path+'/client_3.h5')
		self.models_dict['client_4'].save(self.path+'/client_4.h5')
		self.save_log(logger)


	def parallel_training_medium(self, split_layer, epochs):
		logger = pd.DataFrame()
		metrics_seq = []
		for e in range(epochs):
			print("Start of epoch %d" %(e)) 
			step=0

			# Training
			for row0, row1, row2, row3, row4 in zip_longest(self.train_datasets['client_0'], self.train_datasets['client_1'], self.train_datasets['client_2'], self.train_datasets['client_3'], self.train_datasets['client_4']):
				step+=1
				if row0:
					train_loss = self.train_step(row0[0], row0[1], 'client_0')
				if row1:
					train_loss = self.train_step(row1[0], row1[1], 'client_1')
				if row2:
					train_loss = self.train_step(row2[0], row2[1], 'client_2')
				if row3:
					train_loss = self.train_step(row3[0], row3[1], 'client_3')
				if row4:
					train_loss = self.train_step(row4[0], row4[1], 'client_4')

				# Weights exchange every 50 batches (half-way of Coarse and Fine data sharing granurality)
				if step%50==0:
					server_weights = self.average_weights(split_layer)
					self.update_server_weights(server_weights, split_layer)
					template = 'TRAINING: Epoch {}, Step {}, AUC train: {}, AUC_Mean: {}, AUC train Card: {}'
					print(template.format(e, step, self.metrics['auc_train'].result().numpy(), self.metrics['mean_auc_train'].result().numpy(), self.metrics['auc_train_card'].result().numpy()))

			# Saving training metrics on tensorboard
			with self.tb_writer_t.as_default():
				for met in dict(list(self.metrics.items())[:7]).items():
					tf.summary.scalar(met[0], met[1].result(), step=e)

			# Validation
			for client in self.client_list:
				for step, row in enumerate(self.val_datasets):
					val_loss = self.validation_step(row[0], row[1], client)
					if step%50==0:
						template = 'VALIDATION {}: Epoch {}, Step {}, AUC valid: {}, AUC_Mean: {}, AUC valid Card: {}'
						print(template.format(client, e+1, step, self.metrics['auc_valid'].result().numpy(), self.metrics['mean_auc_valid'].result().numpy(), self.metrics['auc_valid_card'].result().numpy()))

			# Saving validation metrics on tensorboard
			with self.tb_writer_v.as_default():
				for met in dict(list(self.metrics.items())[7:]).items():
					tf.summary.scalar(met[0], met[1].result(), step=e)

			# Log metrics 
			logger = self.log_epoch(logger, e, run='training')
			print(logger)

			# Check for EarlyStopping
			metrics_seq.append(self.metrics['mean_auc_valid'].result().numpy())
			stopEarly = self.callback_earlyStopping(metrics_seq, min_delta=0.1, patience=4, mode='max')
			if stopEarly:
				print("Callback_EarlyStopping signal received at epoch = %d/%d"%(e+1,epochs))
				print("Terminating training ")
				break

			# Reset metrics at each epoch
			for m in self.metrics.values():
				m.reset_states()

		# Save Models and Logs 
		self.models_dict['client_0'].save(self.path+'/client_0.h5')
		self.models_dict['client_1'].save(self.path+'/client_1.h5')
		self.models_dict['client_2'].save(self.path+'/client_2.h5')
		self.models_dict['client_3'].save(self.path+'/client_3.h5')
		self.models_dict['client_4'].save(self.path+'/client_4.h5')		
		self.save_log(logger)
