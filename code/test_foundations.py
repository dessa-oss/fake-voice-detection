import numpy as np
import foundations

model_params = {'num_freq_bin': 240,
                'num_conv_blocks': 8,
                'num_conv_filters': 32,
                'spatial_dropout_fraction': 0.05,
                'num_dense_layers': 1,
                'num_dense_neurons': 50,
                'dense_dropout': 0,
                'learning_rate': 0.0001,
                'epochs': 100,
                'batch_size': 156,
                'residual_con': 2,
                'use_default': True,
                'model_save_dir': 'fitted_objects'
                }

for k, v in model_params.items():
    foundations.log_param(k, v)

train_accuracy = np.random.rand()
foundations.log_metric("train_accuracy", train_accuracy)
foundations.log_metric("val_accuracy", train_accuracy*0.85)

# foundations.save_artifact('visualize_inference_spectrogram.png', key='spectrogram')