import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from ..models.tensorflow import create_model
from ..utils.helpers import create_scaler, split
from ..utils.helpers_tensorflow import create_data_loader, create_learning_rate_scheduler, create_adam_optimizer

def parse_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', metavar='path', type=str, required=True, help='Input HDF5 file containing training data set')
    parser.add_argument('--metrics_file', metavar='path', type=str, required=True, help='Path and name of HDF5 file to store training metrics')
    parser.add_argument('--network_file', metavar='path', type=str, required=True, help='Path and name of HDF5 file to store training metrics')
    parser.add_argument('--input_vars', metavar='vars', type=str, nargs='*', required=True, help='Name(s) of input variables in training data set')
    parser.add_argument('--output_vars', metavar='vars', type=str, nargs='*', required=True, help='Name(s) of output variables in training data set')
    parser.add_argument('--validation_fraction', metavar='frac', type=float, default=0.1, help='Fraction of data set to reserve as validation set')
    parser.add_argument('--test_fraction', metavar='frac', type=float, default=0.1, help='Fraction of data set to reserve as test set')
    parser.add_argument('--shuffle_seed', metavar='seed', type=int, default=None, help='Set the random seed to be used for shuffling')
    parser.add_argument('--sample_seed', metavar='seed', type=int, default=None, help='Set the random seed to be used for OOD sampling')
    parser.add_argument('--hidden_nodes', metavar='n', type=int, default=20, help='Number of nodes in the common hidden layer')
    parser.add_argument('--specialized_nodes', metavar='n', type=int, nargs='*', default=None, help='Number of nodes in the specialized hidden layer')
    parser.add_argument('--batch_size', metavar='n', type=int, default=None, help='Size of minibatch to use in training loop')
    parser.add_argument('--max_epochs', metavar='n', type=int, default=10000, help='Maximum number of epochs to train BNN')
    parser.add_argument('--early_stopping', metavar='patience', type=int, default=None, help='Set number of epochs meeting the criteria needed to trigger early stopping')
    parser.add_argument('--epi_prior', metavar='val', type=float, nargs='*', default=None, help='Standard deviation of epistemic priors used to compute epistemic loss term')
    parser.add_argument('--alea_prior', metavar='val', type=float, nargs='*', default=None, help='Standard deviation of aleatoric priors used to compute aleatoric loss term')
    parser.add_argument('--nll_weight', metavar='wgt', type=float, nargs='*', default=None, help='Weight to apply to the NLL loss term')
    parser.add_argument('--epi_weight', metavar='wgt', type=float, nargs='*', default=None, help='Weight to apply to epistemic loss term')
    parser.add_argument('--alea_weight', metavar='wgt', type=float, nargs='*', default=None, help='Weight to apply to aleatoric loss term')
    parser.add_argument('--learning_rate', metavar='rate', type=float, default=0.001, help='Initial learning rate for Adam optimizer')
    parser.add_argument('--decay_rate', metavar='rate', type=float, default=0.98, help='Scheduled learning rate decay for Adam optimizer')
    parser.add_argument('--decay_epochs', metavar='n', type=float, default=20, help='Epochs between applying learning rate decay for Adam optimizer')
    return parser.parse_args()


def preprocess_data(
    data,
    feature_vars,
    target_vars,
    validation_fraction,
    test_fraction,
    shuffle=True,
    seed=None
):

    feature_scaler = create_scaler(data.loc[:, feature_vars])
    target_scaler = create_scaler(data.loc[:, target_vars])

    ml_vars = []
    ml_vars.extend(feature_vars)
    ml_vars.extend(target_vars)
    ml_data = data.loc[:, ml_vars]

    first_split = validation_fraction + test_fraction
    second_split = test_fraction / first_split
    train_data, split_data = split(ml_data, first_split, shuffle=shuffle, seed=seed)
    val_data, test_data = split(split_data, second_split, shuffle=shuffle, seed=seed)

    feature_train = feature_scaler.transform(train_data.loc[:, feature_vars])
    feature_val = feature_scaler.transform(val_data.loc[:, feature_vars])
    feature_test = feature_scaler.transform(test_data.loc[:, feature_vars])

    target_train = target_scaler.transform(train_data.loc[:, target_vars])
    target_val = target_scaler.transform(val_data.loc[:, target_vars])
    target_test = target_scaler.transform(test_data.loc[:, target_vars])

    features = {
        'names': feature_vars,
        'original': train_data.loc[:, feature_vars].to_numpy(),
        'train': feature_train,
        'validation': feature_val,
        'test': feature_test,
        'scaler': feature_scaler,
    }
    targets = {
        'names': target_vars,
        'original': train_data.loc[:, target_vars].to_numpy(),
        'train': target_train,
        'validation': target_val,
        'test': target_test,
        'scaler': target_scaler,
    }

    return features, targets


@tf.function
def ncp_train_step(
    model,
    optimizer,
    dataloader,
    epi_priors,
    alea_priors,
    nll_weights,
    epi_weights,
    alea_weights,
    ood_sigma,
    ood_seed=None
):

    batch_total_loss = []
    batch_nll_loss = []
    batch_epi_loss = []
    batch_alea_loss = []

    # Loop through minibatches
    for feature_batch, target_batch in dataloader:

        # Define epistemic prior for NCP methodology
        model_priors = []
        for ii in range(len(epi_priors)):
            prior = tfd.Normal(target_batch[:, ii], epi_priors[ii])
            model_priors.append(prior)

        # Define aleatoric prior for NCP methodology
        noise_priors = []
        for ii in range(len(alea_priors)):
            prior = tfd.Normal(target_batch[:, ii], alea_priors[ii])
            noise_priors.append(prior)
        
        # Generate random OOD data from training data
        ood_batch_vectors = []
        for jj in range(len(ood_sigma)):
            val = feature_batch[:, jj]
            ood = val + tf.random.normal((val.shape, ), stddev=ood_sigma[jj], seed=ood_seed)
            ood_batch_vectors.append(ood)
        ood_feature_batch = tf.data.Dataset.from_tensor_slices(ood_batch_vectors)

        with tf.GradientTape() as tape:

            # For mean data inputs, e.g. training data
            mean_outputs = model(feature_batch, training=True)
            mean_model_dists = outputs[::2]
            mean_noise_dists = outputs[1::2]

            # For OOD data inputs
            ood_outputs = model(ood_feature_batch, training=True)
            ood_model_dists = ood_outputs[::2]
            ood_noise_dists = ood_outputs[1::2]

            # Container for total loss
            total_loss = tf.constant(0, shape=(batch_size, 1))

            # Negative log-likelihood loss term: compare probability of target against mean noise distribution
            nll_bases = [np.nan] * len(mean_noise_dists)
            nll_terms = [np.nan] * len(mean_noise_dists)
            for ii in range(len(mean_noise_dists)):
                nll = -noise_dists[ii].log_prob(target_batch[:, ii])
                nll_base = tf.reshape(nll, [batch_size, 1])
                nll_bases[ii] = tf.reduce_sum(nll_base).numpy()
                nll_terms[ii] = tf.reduce_sum(nll_weights[ii] * nll_base).numpy()
                total_loss = total_loss + nll_weights[ii] * nll_base

            # Epistemic loss term: KL-divergence between model prior and OOD model distribution
            epi_bases = [np.nan] * len(ood_model_dists)
            epi_terms = [np.nan] * len(ood_model_dists)
            for ii in range(len(ood_model_dists)):
                kl_ood_model = tfd.kl_divergence(model_priors[ii], ood_model_dists[ii])
                epi_base = tf.reshape(kl_ood_model, [batch_size, 1])
                epi_bases[ii] = tf.reduce_sum(epi_base).numpy()
                epi_terms[ii] = tf.reduce_sum(epi_weights[ii] * epi_base).numpy()
                total_loss = total_loss + epi_weights[ii] * epi_base

            # Aleatoric loss term: KL-divergence between noise prior and OOD noise distribution
            alea_bases = [np.nan] * len(ood_noise_dists)
            alea_terms = [np.nan] * len(ood_noise_dists)
            for ii in range(len(ood_noise_dists)):
                kl_ood_noise = tfd.kl_divergence(noise_priors[ii], ood_noise_dists[ii])
                alea_base = tf.reshape(kl_ood_noise, [batch_size, 1])
                alea_bases[ii] = tf.reduce_sum(alea_base).numpy()
                alea_terms[ii] = tf.reduce_sum(alea_weights[ii] * alea_base).numpy()
                total_loss = total_loss + alea_weights[ii] * alea_base

        # Calculate the combined loss term
        batch_nll_loss.append(np.array(nll_terms))
        batch_epi_loss.append(np.array(epi_terms))
        batch_alea_loss.append(np.array(alea_terms))
        batch_total_loss.append(total_loss)

        # Apply back-propagation
        trainable_vars = model.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))

    nll_losses = np.vstack(batch_nll_loss)
    epi_losses = np.vstack(batch_epi_loss)
    alea_losses = np.vstack(batch_alea_loss)
    total_losses = np.array(batch_total_loss)

    return total_losses, nll_losses, epi_losses, alea_losses


def train(
    model,
    optimizer,
    features_train,
    targets_train,
    features_valid,
    targets_valid,
    max_epochs,
    epi_priors,
    alea_priors,
    nll_weights,
    epi_weights,
    alea_weights,
    batch_size=None,
    patience=None,
    seed=None
):

    n_inputs = features_train.shape[1]
    n_outputs = targets_train.shape[1]
    train_length = features_train.shape[0]
    #valid_length = features_valid.shape[0]

    train_loader = create_data_loader(features_train, targets_train, buffer_size=train_length, seed=seed, batch_size=batch_size)
    valid_loader = create_data_loader(features_valid, targets_valid)

    total_tracker = tf.keras.metrics.Sum(name=f'total')
    nll_trackers = []
    epistemic_trackers = []
    aleatoric_trackers = []
    mae_trackers = []
    mse_trackers = []
    for ii in range(n_outputs):
        nll_trackers.append(tf.keras.metrics.Sum(name=f'nll{ii}'))
        epistemic_trackers.append(tf.keras.metrics.Sum(name=f'epistemic{ii}'))
        aleatoric_trackers.append(tf.keras.metrics.Sum(name=f'aleatoric{ii}'))
        mae_trackers.append(tf.keras.metrics.MeanAbsoluteError(name=f'mae{ii}'))
        mse_trackers.append(tf.keras.metrics.MeanSquaredError(name=f'mse{ii}'))

    total_list = []
    nll_list = []
    epi_list = []
    alea_list = []
    mae_list = []
    mse_list = []

    for epoch in range(max_epochs):

        total, nll, epi, alea = ncp_train_step(
            model,
            optimizer,
            train_loader,
            epi_priors,
            alea_priors,
            nll_weights,
            epi_weights,
            alea_weights,
            ood_sigma,
            ood_seed=None
        )

        train_outputs = model(feature_train, training=False)
        train_model_dists = train_outputs[::2]
        train_noise_dists = train_outputs[1::2]

        total_tracker.update_state(total)
        for ii in range(n_outputs):
            nll_trackers[ii].update_state(nll[:, ii])
            epistemic_trackers[ii].update_state(epi[:, ii])
            aleatoric_trackers[ii].update_state(alea[:, ii])
            mae_trackers[ii].update_state(targets_train[:, ii], train_model_dists[ii].mean())
            mse_trackers[ii].update_state(targets_train[:, ii], train_model_dists[ii].mean())
        
        nll = [np.nan] * n_outputs
        epi = [np.nan] * n_outputs
        alea = [np.nan] * n_outputs
        mae = [np.nan] * n_outputs
        mse = [np.nan] * n_outputs
        for ii in range(n_outputs):
            nll[ii] = nll_trackers[ii].result().numpy()
            epi[ii] = epistemic_trackers[ii].result().numpy()
            alea[ii] = aleatoric_trackers[ii].result().numpy()
            mae[ii] = mae_trackers[ii].result().numpy()
            mse[ii] = mse_trackers[ii].result().numpy()

        total_list.append(total_tracker.result().numpy())
        nll_list.append(nll)
        epi_list.append(epi)
        alea_list.append(alea)
        mae_list.append(mae)
        mse_list.append(mse)

        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}: total = {total_list[-1]:.3f}')
            for ii in range(n_outputs):
                print(f'  Output {ii}: mse = {mse_list[-1][ii]:.3f}, mae = {mae_list[-1][ii]:.3f}, nll = {nll_list[-1][ii]:.3f}, epi = {epi_list[-1][ii]:.3f}, alea = {alea_list[-1][ii]:.3f}')

        total_tracker.reset_states()
        for ii in range(n_outputs):
            nll_trackers[ii].reset_states()
            epistemic_trackers[ii].reset_states()
            aleatoric_trackers[ii].reset_states()
            mae_trackers[ii].reset_states()
            mse_trackers[ii].reset_states()

        if isinstance(patience, int):

            valid_outputs = model(feature_valid, training=False)
            valid_model_dists = valid_outputs[::2]
            valid_noise_dists = valid_outputs[1::2]

    return total_list, mse_list, mae_list, nll_list, epi_list, alea_list


def main():

    args = parse_inputs()

    ipath = Path(args.data_file)
    if ipath.is_file():

        # Set up the required data sets
        start_preprocess = time.perf_counter()
        data = pd.read_hdf(ipath, '/data')
        features, targets = preprocess_data(
            data,
            args.feature_vars,
            args.target_vars,
            args.validation_fraction,
            args.test_fraction,
            seed=args.shuffle_seed
        )
        end_preprocess = time.perf_counter()

        print(f'Pre-processing completed! Elpased time: {(end_preprocess - start_preprocess):.4f} s')

        # Set up the BNN-NCP model
        start_setup = time.perf_counter()
        n_inputs = features['train'].shape[1]
        n_outputs = targets['train'].shape[1]
        model = create_model(
            n_inputs=n_inputs,
            n_hidden=args.hidden_nodes,
            n_outputs=n_outputs,
            n_specialized=args.specialized_nodes
        )

        # Set up the training settings
        epi_priors = [0.001] * n_outputs
        for ii in range(n_outputs):
            if isinstance(args.epi_prior, list):
                epi_priors[ii] = args.epi_prior[ii] if ii < len(args.epi_prior) else args.epi_prior[-1]
            epi_priors[ii] = epi_priors[ii] * features['original'][:, ii]
        alea_priors = [0.001] * n_outputs
        for ii in range(n_outputs):
            if isinstance(args.alea_prior, list):
                alea_priors[ii] = args.alea_prior[ii] if ii < len(args.alea_prior) else args.alea_prior[-1]
            alea_priors[ii] = alea_priors[ii] * features['original'][:, ii]
        nll_weights = [1.0] * n_outputs
        for ii in range(n_outputs):
            if isinstance(args.nll_weight, list):
                nll_weights[ii] = args.nll_weight[ii] if ii < len(args.nll_weight) else args.nll_weight[-1]
        epi_weights = [1.0] * n_outputs
        for ii in range(n_outputs):
            if isinstance(args.epi_weight, list):
                epi_weights[ii] = args.epi_weight[ii] if ii < len(args.epi_weight) else args.epi_weight[-1]
        alea_weights = [1.0] * n_outputs
        for ii in range(n_outputs):
            if isinstance(args.alea_weight, list):
                alea_weights[ii] = args.alea_weight[ii] if ii < len(args.alea_weight) else args.alea_weight[-1]

        train_length = features['train'].shape[0]
        steps_per_epoch = int(np.ceil(train_length / args.batch_size)) if isinstance(args.batch_size, int) else 1
        steps = steps_per_epoch * args.decay_epochs
        scheduler = create_learning_rate_scheduler(
            initial_lr=args.learning_rate,
            decay_steps=steps,
            decay_rate=args.decay_rate
        )
        optimizer = create_adam_optimizer(scheduler)
        end_setup = time.perf_counter()

        print(f'Setup completed! Elapsed time: {(end_setup - start_setup):.4f} s')

        # Perform the training loop
        start_train = time.perf_counter()
        total_list, mse_list, mae_list, nll_list, epistemic_list, aleatoric_list = train(
            model,
            optimizer,
            features['train'],
            targets['train'],
            features['valid'],
            targets['valid'],
            args.max_epochs,
            epi_priors,
            alea_priors,
            nll_weights,
            epi_weights,
            alea_weights,
            batch_size=args.batch_size,
            patience=args.early_stopping,
            seed=args.sampling_seed
        )
        end_train = time.perf_counter()

        print(f'Training loop completed! Elapsed time: {(end_train - start_train):.4f} s')

        # Save the trained model and training metrics
        start_save = time.perf_counter()
        total = np.atleast_2d(total_list)
        mse = np.atleast_2d(mse_list)
        mae = np.atleast_2d(mae_list)
        nll = np.atleast_2d(nll_list)
        epistemic = np.atleast_2d(epistemic_list)
        aleatoric = np.atleast_2d(aleatoric_list)
        metric_dict = {'total': total.flatten()}
        for ii in range(n_outputs):
            metric_dict[f'mse{ii}'] = mse[:, ii].flatten()
            metric_dict[f'mae{ii}'] = mae[:, ii].flatten()
            metric_dict[f'nll{ii}'] = nll[:, ii].flatten()
            metric_dict[f'epi{ii}'] = epistemic[:, ii].flatten()
            metric_dict[f'alea{ii}'] = aleatoric[:, ii].flatten()
        metrics = pd.DataFrame(data=metric_dict)
        metrics.to_hdf(mpath, '/data')
        model.save(npath)
        end_save = time.perf_counter()

        print(f'Saving completed! Elapsed time: {(end_save - start_save):.4f} s')

        print(f'Script completed: Network saved in {npath}, metrics saved in {mpath}')

    else:
        raise IOError(f'Could not find input file: {ipath}')


if __name__ == "__main__":
    main()
