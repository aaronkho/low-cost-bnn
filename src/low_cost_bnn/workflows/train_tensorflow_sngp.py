import argparse
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from ..utils.pipeline_tools import setup_logging, print_settings, preprocess_data
from ..utils.helpers_tensorflow import default_dtype, set_tf_logging_level, create_data_loader, create_scheduled_adam_optimizer, create_classifier_model, create_classifier_loss_function, wrap_classifier_model, save_model

logger = logging.getLogger("train_tensorflow")


def parse_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', metavar='path', type=str, required=True, help='Path and name of input HDF5 file containing training data set')
    parser.add_argument('--metrics_file', metavar='path', type=str, required=True, help='Path and name of output HDF5 file to store training metrics')
    parser.add_argument('--network_file', metavar='path', type=str, required=True, help='Path and name of output file to store training metrics')
    parser.add_argument('--input_var', metavar='vars', type=str, nargs='*', required=True, help='Name(s) of input variables in training data set')
    parser.add_argument('--output_var', metavar='vars', type=str, nargs='*', required=True, help='Name(s) of output variables in training data set')
    parser.add_argument('--validation_fraction', metavar='frac', type=float, default=0.1, help='Fraction of data set to reserve as validation set')
    parser.add_argument('--test_fraction', metavar='frac', type=float, default=0.1, help='Fraction of data set to reserve as test set')
    parser.add_argument('--test_file', metavar='path', type=str, default=None, help='Optional path to output HDF5 file where test partition will be saved')
    parser.add_argument('--max_epoch', metavar='n', type=int, default=100000, help='Maximum number of epochs to train BNN')
    parser.add_argument('--batch_size', metavar='n', type=int, default=None, help='Size of minibatch to use in training loop')
    parser.add_argument('--early_stopping', metavar='patience', type=int, default=None, help='Set number of epochs meeting the criteria needed to trigger early stopping')
    parser.add_argument('--shuffle_seed', metavar='seed', type=int, default=None, help='Set the random seed to be used for shuffling')
    parser.add_argument('--generalized_node', metavar='n', type=int, nargs='*', default=None, help='Number of nodes in the generalized hidden layers')
    parser.add_argument('--specialized_layer', metavar='n', type=int, nargs='*', default=None, help='Number of specialized hidden layers, given for each output')
    parser.add_argument('--specialized_node', metavar='n', type=int, nargs='*', default=None, help='Number of nodes in the specialized hidden layers, sequential per output stack')
    parser.add_argument('--spec_norm_general', metavar='wgt', type=float, default=0.9, help='Spectral normalization parameter used in the generalized hidden layers')
    parser.add_argument('--rel_norm_special', metavar='wgt', type=float, default=1.0, help='Relative spectral normalization used in the specialized hidden layers compared to the generalized layers')
    parser.add_argument('--entropy_weight', metavar='wgt', type=float, default=1.0, help='Weight to apply to the cross-entropy loss term')
    parser.add_argument('--reg_weight', metavar='wgt', type=float, default=1.0, help='Weight to apply to regularization loss term (not applicable here)')
    parser.add_argument('--n_class', metavar='n', type=int, default=1, help='Total number of possible classes present in classification target data')
    parser.add_argument('--learning_rate', metavar='rate', type=float, default=0.001, help='Initial learning rate for Adam optimizer')
    parser.add_argument('--decay_rate', metavar='rate', type=float, default=0.98, help='Scheduled learning rate decay for Adam optimizer')
    parser.add_argument('--decay_epoch', metavar='n', type=float, default=50, help='Epochs between applying learning rate decay for Adam optimizer')
    parser.add_argument('--disable_gpu', default=False, action='store_true', help='Toggle off GPU usage provided that GPUs are available on the device')
    parser.add_argument('--log_file', metavar='path', type=str, default=None, help='Optional path to output log file where script related print outs will be stored')
    parser.add_argument('-v', dest='verbosity', action='count', default=0, help='Set level of verbosity for the training script')
    return parser.parse_args()


@tf.function
def train_tensorflow_sngp_epoch(
    model,
    optimizer,
    dataloader,
    loss_function,
    reg_weight,
    training=True,
    dataset_length=None,
    verbosity=0
):

    # Using the None option here is unwieldy for large datasets, recommended to always pass in correct length
    dataset_size = tf.cast(dataloader.unbatch().cardinality(), dtype=default_dtype) if dataset_length is None else tf.constant(dataset_length, dtype=default_dtype)

    step_total_losses = tf.TensorArray(dtype=default_dtype, size=0, dynamic_size=True, clear_after_read=True, name=f'total_loss_array')
    step_entropy_losses = tf.TensorArray(dtype=default_dtype, size=0, dynamic_size=True, clear_after_read=True, name=f'entropy_loss_array')

    # Custom model function resets the covariance matrix, critical for proper training of SNGP architecture
    if hasattr(model, 'pre_epoch_processing'):
        model.pre_epoch_processing()

    # Training loop through minibatches - each loop pass is one step
    for nn, (feature_batch, target_batch) in enumerate(dataloader):

        batch_size = tf.cast(tf.shape(feature_batch)[0], dtype=default_dtype)

        # Set up training targets into a single large tensor
        batch_loss_targets = target_batch
        n_classes = model.n_outputs

        with tf.GradientTape() as tape:

            # For mean data inputs, e.g. training data
            outputs = model(feature_batch, training=training)

            if training and tf.executing_eagerly() and verbosity >= 4:
                for ii in range(n_classes):
                    logger.debug(f'     logit {ii}: {outputs[0, 0, ii]}')
                    logger.debug(f'     variance {ii}: {outputs[0, 1, ii]}')

            batch_loss_predictions = tf.squeeze(tf.gather(outputs, indices=[0], axis=1), axis=1)

            # Compute total loss to be used in adjusting weights and biases
            step_total_loss = loss_function(batch_loss_targets, batch_loss_predictions)
            adjusted_step_total_loss = tf.math.divide(step_total_loss, batch_size)
            step_entropy_loss = step_total_loss

        # Apply back-propagation
        if training:
            trainable_vars = model.trainable_variables
            gradients = tape.gradient(adjusted_step_total_loss, trainable_vars)
            optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Accumulate batch losses to determine epoch loss
        fill_index = tf.cast(nn + 1, tf.int32)
        step_total_losses = step_total_losses.write(fill_index, tf.reshape(step_total_loss, shape=[-1, 1]))
        step_entropy_losses = step_entropy_losses.write(fill_index, tf.reshape(step_entropy_loss, shape=[-1, 1]))

        if tf.executing_eagerly() and verbosity >= 3:
            if training:
                logger.debug(f'  - Batch {nn + 1}: total = {step_total_loss:.3f}, entropy = {step_entropy_loss:.3f}')
            else:
                logger.debug(f'  - Validation: total = {step_total_loss:.3f}, entropy = {step_entropy_loss:.3f}')

    epoch_total_loss = tf.reduce_sum(step_total_losses.concat(), axis=0)
    epoch_entropy_loss = tf.reduce_sum(step_entropy_losses.concat(), axis=0)

    return epoch_total_loss, epoch_entropy_loss


def train_tensorflow_sngp(
    model,
    optimizer,
    features_train,
    targets_train,
    features_valid,
    targets_valid,
    loss_function,
    reg_weight,
    max_epochs,
    batch_size=None,
    patience=None,
    seed=None,
    verbosity=0
):

    n_inputs = features_train.shape[-1]
    n_outputs = targets_train.shape[-1]
    train_length = features_train.shape[0]
    valid_length = features_valid.shape[0]
    n_no_improve = 0
    improve_tol = 0.0
    roc_thresholds = np.linspace(0.0, 1.0, 101).tolist()[1:-1]
    idx_def = 50
    beta = 1.0
    optimal_class_ratio = 0.5

    if verbosity >= 2:
        logger.info(f' Number of inputs: {n_inputs}')
        logger.info(f' Number of outputs: {n_outputs}')
        logger.info(f' Training set size: {train_length}')
        logger.info(f' Validation set size: {valid_length}')

    # Create data loaders, including minibatching for training set
    train_data = (features_train.astype(default_dtype), targets_train.astype(default_dtype))
    valid_data = (features_valid.astype(default_dtype), targets_valid.astype(default_dtype))
    train_loader = create_data_loader(train_data, buffer_size=train_length, seed=seed, batch_size=batch_size)
    valid_loader = create_data_loader(valid_data, batch_size=valid_length)

    # Create training tracker objects to facilitate external analysis of pipeline
    multi_label = (n_outputs > 1)
    total_train_tracker = tf.keras.metrics.Sum(name=f'train_total')
    entropy_train_tracker = tf.keras.metrics.Sum(name=f'train_entropy')
    #f1_train_tracker = tf.keras.metrics.F1Score(threshold=0.5, name=f'train_f1')
    auc_train_trackers = []
    tp_train_trackers = []
    tn_train_trackers = []
    fp_train_trackers = []
    fn_train_trackers = []
    for ii in range(n_outputs):
        auc_train_trackers.append(tf.keras.metrics.AUC(num_thresholds=101, name=f'train_auc{ii}'))
        tp_train_trackers.append(tf.keras.metrics.TruePositives(thresholds=roc_thresholds, name=f'train_tp{ii}'))
        tn_train_trackers.append(tf.keras.metrics.TrueNegatives(thresholds=roc_thresholds, name=f'train_tn{ii}'))
        fp_train_trackers.append(tf.keras.metrics.FalsePositives(thresholds=roc_thresholds, name=f'train_fp{ii}'))
        fn_train_trackers.append(tf.keras.metrics.FalseNegatives(thresholds=roc_thresholds, name=f'train_fn{ii}'))

    # Create validation tracker objects to facilitate external analysis of pipeline
    total_valid_tracker = tf.keras.metrics.Sum(name=f'valid_total')
    entropy_valid_tracker = tf.keras.metrics.Sum(name=f'valid_entropy')
    #f1_valid_tracker = tf.keras.metrics.F1Score(threshold=0.5, name=f'valid_f1')
    auc_valid_trackers = []
    tp_valid_trackers = []
    tn_valid_trackers = []
    fp_valid_trackers = []
    fn_valid_trackers = []
    for ii in range(n_outputs):
        auc_valid_trackers.append(tf.keras.metrics.AUC(num_thresholds=101, name=f'valid_auc{ii}'))
        tp_valid_trackers.append(tf.keras.metrics.TruePositives(thresholds=roc_thresholds, name=f'valid_tp{ii}'))
        tn_valid_trackers.append(tf.keras.metrics.TrueNegatives(thresholds=roc_thresholds, name=f'valid_tn{ii}'))
        fp_valid_trackers.append(tf.keras.metrics.FalsePositives(thresholds=roc_thresholds, name=f'valid_fp{ii}'))
        fn_valid_trackers.append(tf.keras.metrics.FalseNegatives(thresholds=roc_thresholds, name=f'valid_fn{ii}'))

    # Output metrics containers
    total_train_list = []
    entropy_train_list = []
    #f1_train_list = []
    auc_train_list = []
    tp_train_list = []
    tn_train_list = []
    fp_train_list = []
    fn_train_list = []
    fb_train_list = []
    thr_train_list = []
    total_valid_list = []
    entropy_valid_list = []
    #f1_valid_list = []
    auc_valid_list = []
    tp_valid_list = []
    tn_valid_list = []
    fp_valid_list = []
    fn_valid_list = []
    fb_valid_list = []
    thr_valid_list = []

    # Output container for the best trained model
    best_validation_loss = None
    best_model = tf.keras.models.clone_model(model)
    best_model.set_weights(model.get_weights())

    # Training loop
    stop_requested = False
    for epoch in range(max_epochs):

        # Training routine described in here
        epoch_total, epoch_entropy = train_tensorflow_sngp_epoch(
            model,
            optimizer,
            train_loader,
            loss_function,
            reg_weight,
            training=True,
            dataset_length=train_length,
            verbosity=verbosity
        )

        # Evaluate model with full training data set for performance tracking
        train_outputs = model(train_data[0], training=False)
        train_means = tf.squeeze(tf.gather(train_outputs, indices=[0], axis=1), axis=1)
        train_vars = tf.squeeze(tf.gather(train_outputs, indices=[1], axis=1), axis=1)
        train_probs = tf.math.sigmoid(train_means / tf.sqrt(1.0 + (tf.math.acos(1.0) / 8.0) * train_vars))

        total_train_tracker.update_state(epoch_total / train_length)
        entropy_train_tracker.update_state(epoch_entropy / train_length)
        #f1_train_tracker.update_state(train_data[1], train_probs.numpy())
        for ii in range(n_outputs):
            metric_targets = train_data[1][:, ii]
            metric_results = train_probs[:, ii].numpy()
            auc_train_trackers[ii].update_state(metric_targets, metric_results)
            tp_train_trackers[ii].update_state(metric_targets, metric_results)
            tn_train_trackers[ii].update_state(metric_targets, metric_results)
            fp_train_trackers[ii].update_state(metric_targets, metric_results)
            fn_train_trackers[ii].update_state(metric_targets, metric_results)

        total_train = total_train_tracker.result().numpy().tolist()
        entropy_train = [np.nan] * n_outputs
        #f1_train = [np.nan] * n_outputs
        auc_train = [np.nan] * n_outputs
        tp_opt_train = [np.nan] * n_outputs
        tn_opt_train = [np.nan] * n_outputs
        fp_opt_train = [np.nan] * n_outputs
        fn_opt_train = [np.nan] * n_outputs
        fb_opt_train = [np.nan] * n_outputs
        thr_opt_train = [np.nan] * n_outputs
        for ii in range(n_outputs):
            entropy_train[ii] = entropy_train_tracker.result().numpy().tolist()
            #f1_train[ii] = f1_train_tracker.result()[ii].numpy().tolist()
            auc_train[ii] = auc_train_trackers[ii].result().numpy().tolist()
            tp_curve_train = tp_train_trackers[ii].result().numpy()
            tn_curve_train = tn_train_trackers[ii].result().numpy()
            fp_curve_train = fp_train_trackers[ii].result().numpy()
            fn_curve_train = fn_train_trackers[ii].result().numpy()
            fb_curve_train = (1.0 + beta ** 2.0) * tp_curve_train / ((1.0 + beta ** 2.0) * tp_curve_train + (beta ** 2.0) * fn_curve_train + fp_curve_train)
            opt_index = np.argmax(fb_curve_train)
            tp_opt_train[ii] = tp_curve_train.tolist()[opt_index]
            tn_opt_train[ii] = tn_curve_train.tolist()[opt_index]
            fp_opt_train[ii] = fp_curve_train.tolist()[opt_index]
            fn_opt_train[ii] = fn_curve_train.tolist()[opt_index]
            fb_opt_train[ii] = fb_curve_train.tolist()[opt_index]
            thr_opt_train[ii] = roc_thresholds[opt_index]

        total_train_list.append(total_train)
        entropy_train_list.append(entropy_train)
        #f1_train_list.append(f1_train)
        auc_train_list.append(auc_train)
        tp_train_list.append(tp_opt_train)
        tn_train_list.append(tn_opt_train)
        fp_train_list.append(fp_opt_train)
        fn_train_list.append(fn_opt_train)
        fb_train_list.append(fb_opt_train)
        thr_train_list.append(thr_opt_train)

        # Reuse training routine to evaluate validation data
        valid_total, valid_entropy = train_tensorflow_sngp_epoch(
            model,
            optimizer,
            valid_loader,
            loss_function,
            reg_weight,
            training=False,
            dataset_length=valid_length,
            verbosity=verbosity
        )

        # Evaluate model with validation data set for performance tracking
        valid_outputs = model(valid_data[0], training=False)
        valid_means = tf.squeeze(tf.gather(valid_outputs, indices=[0], axis=1), axis=1)
        valid_vars = tf.squeeze(tf.gather(valid_outputs, indices=[1], axis=1), axis=1)
        valid_probs = tf.math.sigmoid(valid_means / tf.sqrt(1.0 + (tf.math.acos(1.0) / 8.0) * valid_vars))

        total_valid_tracker.update_state(valid_total / valid_length)
        entropy_valid_tracker.update_state(valid_entropy / valid_length)
        #f1_valid_tracker.update_state(valid_data[1], valid_probs.numpy())
        for ii in range(n_outputs):
            metric_targets = valid_data[1][:, ii]
            metric_results = valid_probs[:, ii].numpy()
            auc_valid_trackers[ii].update_state(metric_targets, metric_results)
            tp_valid_trackers[ii].update_state(metric_targets, metric_results)
            tn_valid_trackers[ii].update_state(metric_targets, metric_results)
            fp_valid_trackers[ii].update_state(metric_targets, metric_results)
            fn_valid_trackers[ii].update_state(metric_targets, metric_results)

        total_valid = total_valid_tracker.result().numpy().tolist()
        entropy_valid = [np.nan] * n_outputs
        #f1_valid = [np.nan] * n_outputs
        auc_valid = [np.nan] * n_outputs
        tp_opt_valid = [np.nan] * n_outputs
        tn_opt_valid = [np.nan] * n_outputs
        fp_opt_valid = [np.nan] * n_outputs
        fn_opt_valid = [np.nan] * n_outputs
        fb_opt_valid = [np.nan] * n_outputs
        thr_opt_valid = [np.nan] * n_outputs
        for ii in range(n_outputs):
            entropy_valid[ii] = entropy_valid_tracker.result().numpy().tolist()
            #f1_valid[ii] = f1_valid_tracker.result()[ii].numpy().tolist()
            auc_valid[ii] = auc_valid_trackers[ii].result().numpy().tolist()
            tp_curve_valid = tp_valid_trackers[ii].result().numpy()
            tn_curve_valid = tn_valid_trackers[ii].result().numpy()
            fp_curve_valid = fp_valid_trackers[ii].result().numpy()
            fn_curve_valid = fn_valid_trackers[ii].result().numpy()
            fb_curve_valid = (1.0 + beta ** 2.0) * tp_curve_valid / ((1.0 + beta ** 2.0) * tp_curve_valid + (beta ** 2.0) * fn_curve_valid + fp_curve_valid)
            opt_index = np.argmax(fb_curve_valid)
            tp_opt_valid[ii] = tp_curve_valid.tolist()[opt_index]
            tn_opt_valid[ii] = tn_curve_valid.tolist()[opt_index]
            fp_opt_valid[ii] = fp_curve_valid.tolist()[opt_index]
            fn_opt_valid[ii] = fn_curve_valid.tolist()[opt_index]
            fb_opt_valid[ii] = fb_curve_valid.tolist()[opt_index]
            thr_opt_valid[ii] = roc_thresholds[opt_index]

        total_valid_list.append(total_valid)
        entropy_valid_list.append(entropy_valid)
        #f1_valid_list.append(f1_valid)
        auc_valid_list.append(auc_valid)
        tp_valid_list.append(tp_opt_valid)
        tn_valid_list.append(tn_opt_valid)
        fp_valid_list.append(fp_opt_valid)
        fn_valid_list.append(fn_opt_valid)
        fb_valid_list.append(fb_opt_valid)
        thr_valid_list.append(thr_opt_valid)

        # Set optimal thresholds using ROC analysis
        model.set_thresholds([float(val) for val in thr_opt_train])

        # Save model into output container if it is the best so far
        if best_validation_loss is None:
            best_validation_loss = total_valid_list[-1] + improve_tol + 1.0e-3
        n_no_improve = n_no_improve + 1 if best_validation_loss < (total_valid_list[-1] + improve_tol) else 0
        if n_no_improve == 0:
            best_validation_loss = total_valid_list[-1]
            best_model.set_weights(model.get_weights())

        # Request training stop if early stopping is enabled
        if isinstance(patience, int) and patience > 0 and n_no_improve >= patience:
            stop_requested = True

        print_per_epochs = 100
        if verbosity >= 2:
            print_per_epochs = 10
        if verbosity >= 3:
            print_per_epochs = 1
        if (epoch + 1) % print_per_epochs == 0:
            logger.info(f' Epoch {epoch + 1}: total_train = {total_train_list[-1]:.3f}, total_valid = {total_valid_list[-1]:.3f}')
            for ii in range(n_outputs):
                logger.debug(f'  Train: Output {ii}: fb = {fb_train_list[-1][ii]:.3f}, auc = {auc_train_list[-1][ii]:.3f}, threshold = {thr_train_list[-1][ii]:.2f}, entropy = {entropy_train_list[-1][ii]:.3f}')
                logger.debug(f'  Valid: Output {ii}: fb = {fb_valid_list[-1][ii]:.3f}, auc = {auc_valid_list[-1][ii]:.3f}, threshold = {thr_valid_list[-1][ii]:.2f}, entropy = {entropy_valid_list[-1][ii]:.3f}')

        total_train_tracker.reset_states()
        entropy_train_tracker.reset_states()
        #f1_train_tracker.reset_states()
        total_valid_tracker.reset_states()
        entropy_valid_tracker.reset_states()
        #f1_valid_tracker.reset_states()
        for ii in range(n_outputs):
            auc_train_trackers[ii].reset_states()
            tp_train_trackers[ii].reset_states()
            tn_train_trackers[ii].reset_states()
            fp_train_trackers[ii].reset_states()
            fn_train_trackers[ii].reset_states()
            auc_valid_trackers[ii].reset_states()
            tp_valid_trackers[ii].reset_states()
            tn_valid_trackers[ii].reset_states()
            fp_valid_trackers[ii].reset_states()
            fn_valid_trackers[ii].reset_states()

        # Exit training loop early if requested by early stopping
        if stop_requested:
            break

    if stop_requested:
        logger.info(f'Early training loop exit triggered at epoch {epoch + 1}!')
    else:
        logger.info(f'Training loop exited at max epoch {epoch + 1}')

    last_index_to_keep = -n_no_improve if n_no_improve > 0 and stop_requested else None
    metrics_dict = {
        'train_total': total_train_list[:last_index_to_keep],
        'valid_total': total_valid_list[:last_index_to_keep],
        'train_auc': auc_train_list[:last_index_to_keep],
        #'train_f1': f1_train_list[:last_index_to_keep],
        'train_tp': tp_train_list[:last_index_to_keep],
        'train_tn': tn_train_list[:last_index_to_keep],
        'train_fp': fp_train_list[:last_index_to_keep],
        'train_fn': fn_train_list[:last_index_to_keep],
        'train_fbeta': fb_train_list[:last_index_to_keep],
        'train_threshold': thr_train_list[:last_index_to_keep],
        'train_entropy': entropy_train_list[:last_index_to_keep],
        'valid_auc': auc_valid_list[:last_index_to_keep],
        #'valid_f1': f1_valid_list[:last_index_to_keep],
        'valid_tp': tp_valid_list[:last_index_to_keep],
        'valid_tn': tn_valid_list[:last_index_to_keep],
        'valid_fp': fp_valid_list[:last_index_to_keep],
        'valid_fn': fn_valid_list[:last_index_to_keep],
        'valid_fbeta': fb_valid_list[:last_index_to_keep],
        'valid_threshold': thr_valid_list[:last_index_to_keep],
        'valid_entropy': entropy_valid_list[:last_index_to_keep],
    }

    return best_model, metrics_dict


def launch_tensorflow_pipeline_sngp(
    data,
    input_vars,
    output_vars,
    validation_fraction=0.1,
    test_fraction=0.1,
    test_file=None,
    max_epoch=100000,
    batch_size=None,
    early_stopping=None,
    shuffle_seed=None,
    generalized_widths=None,
    specialized_depths=None,
    specialized_widths=None,
    spectral_normalization=0.9,
    relative_normalization=1.0,
    entropy_weights=1.0,
    regularization_weights=1.0,
    total_classes=1,
    learning_rate=0.001,
    decay_epoch=0.98,
    decay_rate=50,
    verbosity=0
):

    settings = {
        'validation_fraction': validation_fraction,
        'test_fraction': test_fraction,
        'test_file': test_file,
        'max_epoch': max_epoch,
        'batch_size': batch_size,
        'early_stopping': early_stopping,
        'shuffle_seed': shuffle_seed,
        'generalized_widths': generalized_widths,
        'specialized_depths': specialized_depths,
        'specialized_widths': specialized_widths,
        'spectral_normalization': spectral_normalization,
        'relative_normalization': relative_normalization,
        'entropy_weights': entropy_weights,
        'regularization_weights': regularization_weights,
        'total_classes': total_classes,
        'learning_rate': learning_rate,
        'decay_epoch': decay_epoch,
        'decay_rate': decay_rate,
    }

    if verbosity >= 1:
        print_settings(logger, settings, 'SNGP model and training settings:')

    # Set up the required data sets
    start_preprocess = time.perf_counter()
    spath = Path(test_file) if isinstance(test_file, str) else None
    features, targets = preprocess_data(
        data,
        input_vars,
        output_vars,
        validation_fraction,
        test_fraction,
        test_savepath=spath,
        seed=shuffle_seed,
        scale_features=True,
        scale_targets=False,
        logger=logger,
        verbosity=verbosity
    )
    if verbosity >= 2:
        logger.debug(f'  Input scaling mean: {features["scaler"].mean_}')
        logger.debug(f'  Input scaling std: {features["scaler"].scale_}')
    end_preprocess = time.perf_counter()

    logger.info(f'Pre-processing completed! Elapsed time: {(end_preprocess - start_preprocess):.4f} s')

    # Set up the SNGP BNN model
    start_setup = time.perf_counter()
    n_inputs = features['train'].shape[-1]
    n_outputs = targets['train'].shape[-1]
    n_commons = len(generalized_widths) if isinstance(generalized_widths, (list, tuple)) else 0
    common_nodes = list(generalized_widths) if n_commons > 0 else None
    special_nodes = None
    if isinstance(specialized_depths, (list, tuple)) and len(specialized_depths) > 0:
        special_nodes = []
        kk = 0
        for jj in range(len(specialized_depths)):
            output_special_nodes = []
            if isinstance(specialized_widths, (list, tuple)) and kk < len(specialized_widths):
                output_special_nodes.append(specialized_widths[kk])
                kk += 1
            special_nodes.append(output_special_nodes)   # List of lists
    model = create_classifier_model(
        n_input=n_inputs,
        n_output=total_classes,
        n_common=n_commons,
        common_nodes=common_nodes,
        special_nodes=special_nodes,
        spectral_norm=spectral_normalization,
        relative_norm=relative_normalization,
        style='sngp',
        verbosity=verbosity
    )

    # Create custom loss function, weights converted into tensor objects internally
    loss_function = create_classifier_loss_function(
        n_outputs,
        style='sngp',
        h_weights=entropy_weights,
        verbosity=verbosity
    )

    train_length = features['train'].shape[0]
    steps_per_epoch = int(np.ceil(train_length / batch_size)) if isinstance(batch_size, int) else 1
    decay_steps = steps_per_epoch * decay_epoch
    optimizer, scheduler = create_scheduled_adam_optimizer(
        model=model,
        learning_rate=learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate
    )
    end_setup = time.perf_counter()

    logger.info(f'Setup completed! Elapsed time: {(end_setup - start_setup):.4f} s')

    # Perform the training loop
    start_train = time.perf_counter()
    best_model, metrics = train_tensorflow_sngp(
        model,
        optimizer,
        features['train'],
        targets['train'],
        features['validation'],
        targets['validation'],
        loss_function,
        regularization_weights,
        max_epoch,
        batch_size=batch_size,
        patience=early_stopping,
        verbosity=verbosity
    )
    end_train = time.perf_counter()

    logger.info(f'Training loop completed! Elapsed time: {(end_train - start_train):.4f} s')

    # Configure the trained model and training metrics for saving
    start_out = time.perf_counter()
    metrics_dict = {}
    for key, val in metrics.items():
        if key.endswith('total') or key.endswith('reg'):
            metric = np.array(val)
            metrics_dict[f'{key}'] = metric.flatten()
        else:
            metric = np.atleast_2d(val)
            for ii in range(n_outputs):
                metrics_dict[f'{key}{ii}'] = metric[:, ii].flatten()
    metrics_df = pd.DataFrame(data=metrics_dict)
    wrapped_model = wrap_classifier_model(best_model, features['scaler'], output_vars)
    end_out = time.perf_counter()

    logger.info(f'Output configuration completed! Elapsed time: {(end_out - start_out):.4f} s')

    if verbosity >= 2:
        inputs = tf.zeros([1, best_model.n_inputs], dtype=default_dtype)
        outputs = model(inputs)
        logger.debug(f'  Sample output at origin:')
        logger.debug(f'{outputs}')

    return wrapped_model, metrics_df


def main():

    args = parse_inputs()

    ipath = Path(args.data_file)
    mpath = Path(args.metrics_file)
    npath = Path(args.network_file)

    if not ipath.is_file():
        raise IOError(f'Could not find input data file: {ipath}')

    if verbosity <= 4:
        set_tf_logging_level(logging.ERROR)

    if args.disable_gpu:
        tf.config.set_visible_devices([], 'GPU')

    if args.verbosity >= 2:
        tf.config.run_functions_eagerly(True)

    lpath = Path(args.log_file) if isinstance(args.log_file, str) else None
    setup_logging(logger, lpath, args.verbosity)
    logger.info(f'Starting SNGP BNN training script...')
    if args.verbosity >= 1:
        print_settings(logger, vars(args), 'SNGP training pipeline CLI settings:')

    start_pipeline = time.perf_counter()

    data = pd.read_hdf(ipath, key='/data')

    trained_model, metrics_dict = launch_tensorflow_pipeline_sngp(
        data=data,
        input_vars=args.input_var,
        output_vars=args.output_var,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        test_file=args.test_file,
        max_epoch=args.max_epoch,
        batch_size=args.batch_size,
        early_stopping=args.early_stopping,
        shuffle_seed=args.shuffle_seed,
        generalized_widths=args.generalized_node,
        specialized_depths=args.specialized_layer,
        specialized_widths=args.specialized_node,
        spectral_normalization=args.spec_norm_general,
        relative_normalization=args.rel_norm_special,
        entropy_weights=args.entropy_weight,
        regularization_weights=args.reg_weight,
        total_classes=args.n_class,
        learning_rate=args.learning_rate,
        decay_epoch=args.decay_epoch,
        decay_rate=args.decay_rate,
        verbosity=args.verbosity
    )

    metrics_dict.to_hdf(mpath, key='/data')
    logger.info(f' Metrics saved in {mpath}')

    save_model(trained_model, npath)
    logger.info(f' Network saved in {npath}')

    end_pipeline = time.perf_counter()

    logger.info(f'Pipeline completed! Total time: {(end_pipeline - start_pipeline):.4f} s')

    logger.info(f'Script completed!')


if __name__ == "__main__":
    main()

