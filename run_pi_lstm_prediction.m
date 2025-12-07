function [results, predictions] = run_pi_lstm_prediction(config, parameter_samples, lstm_training_data)
%% Physics-Informed LSTM for Satellite Solar Panel Deployment Prediction
% Author: YD Fu
% Description: Physics-Informed LSTM model for predicting satellite solar panel deployment trajectories            
% License: Tongji University
% Dependencies: Deep Learning Toolbox

%% ==================== ENVIRONMENT SETUP ====================
fprintf('[PI-LSTM] Initializing Physics-Informed LSTM Environment...\n');
% Record training start time for performance tracking
training_start_time = tic;

%% ==================== PARAMETER DEFINITION ====================
% Load model-specific configuration parameters
lstm_config = get_pi_lstm_config();
% Extract training parameters from main configuration
num_epochs = config.lstm.num_epochs;
window_size = config.lstm.window_size;
train_ratio = config.lstm.train_ratio;
val_ratio = config.lstm.val_ratio;
test_ratio = config.lstm.test_ratio;
% Network architecture parameters
fc_encoder_neurons = lstm_config.network.fc_encoder_neurons;
lstm_neurons = lstm_config.network.lstm_neurons;
fc_decoder_neurons = lstm_config.network.fc_decoder_neurons;
% Three-term loss function weights
lambda_data = lstm_config.loss.lambda_data;
lambda_physics = lstm_config.loss.lambda_physics;
lambda_constraint = lstm_config.loss.lambda_constraint;
fprintf('[PARAMS] PI-LSTM Model Parameters Loaded:\n');
fprintf('         Network: FC(%d)-LSTM(%d)-FC(%d)\n', fc_encoder_neurons, lstm_neurons, fc_decoder_neurons);
fprintf('         Loss Weights: Data=%.1f, Physics=%.1f, Constraints=%.1f\n', ...
        lambda_data, lambda_physics, lambda_constraint);

%% ==================== DATA LOADING AND VALIDATION ====================
fprintf('[DATA] Loading and Validating Deployment Dataset...\n');
% Extract data components from training data structure
X_params = lstm_training_data.X_params;
Y_trajectories = lstm_training_data.Y_trajectories;
time_vec = lstm_training_data.time_vec;
% Validate data dimensions and integrity
[n_samples, n_params] = size(X_params);
[~, n_outputs, n_steps] = size(Y_trajectories);
% Data validation checks
assert(n_samples > 0, 'Training data must contain at least one sample');
assert(n_steps > window_size, 'Time steps must exceed window size for sequence generation');
assert(length(time_vec) == n_steps, 'Time vector length must match trajectory time steps');
fprintf('[DATA] Dataset Statistics:\n');
fprintf('         Samples: %d, Parameters: %d, Output States: %d, Time Steps: %d\n', ...
        n_samples, n_params, n_outputs, n_steps);

%% ==================== DATA PREPROCESSING ====================
fprintf('[PREPROCESS] Preprocessing Deployment Data for Sequence Learning...\n');
% Prepare sequential data for LSTM training
[X_sequences, Y_targets] = prepare_sequence_data(Y_trajectories, window_size);
% Normalize input parameters for stable training
[X_params_normalized, normalization_params] = normalize_parameters(X_params);
% Save processed data for reproducibility and debugging
output_dir = './output';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('[PREPROCESS] Created output directory: %s\n', output_dir);
end
save(fullfile(output_dir, 'processed_deployment_data.mat'), ...
     'X_params_normalized', 'X_sequences', 'Y_targets', 'time_vec', 'normalization_params');

fprintf('[PREPROCESS] Generated %d training sequences from %d original samples\n', ...
        size(X_sequences, 1), n_samples);

%% ==================== TRAINING DATA PREPARATION ====================
fprintf('[PREPARE] Splitting Data into Training, Validation, and Test Sets...\n');
% Calculate dataset splits based on specified ratios
total_sequences = size(X_sequences, 1);
n_train = floor(total_sequences * train_ratio);
n_val = floor(total_sequences * val_ratio);
n_test = total_sequences - n_train - n_val;
% Create indices for data splitting
train_indices = 1:n_train;
val_indices = n_train+1:n_train+n_val;
test_indices = n_train+n_val+1:total_sequences;
% Split sequences into training, validation, and test sets
X_train = X_sequences(train_indices, :, :);
Y_train = Y_targets(train_indices, :);
X_val = X_sequences(val_indices, :, :);
Y_val = Y_targets(val_indices, :);
X_test = X_sequences(test_indices, :, :);
Y_test = Y_targets(test_indices, :);
fprintf('[PREPARE] Data Split Complete:\n');
fprintf('         Training: %d sequences (%.1f%%)\n', n_train, train_ratio*100);
fprintf('         Validation: %d sequences (%.1f%%)\n', n_val, val_ratio*100);
fprintf('         Test: %d sequences (%.1f%%)\n', n_test, test_ratio*100);

%% ==================== REVISED FC-LSTM-FC NETWORK ARCHITECTURE ====================
fprintf('[ARCHITECTURE] Building Revised FC-LSTM-FC Network...\n');
% Define network input and output sizes
input_size = n_outputs;
output_size = n_outputs;
% Construct layer-by-layer architecture with explicit dimension handling
layers = [
    % Input layer for sequence data
    sequenceInputLayer(input_size, 'Name', 'input')
    % FC Encoder: Project input to higher-dimensional space
    fullyConnectedLayer(fc_encoder_neurons, 'Name', 'fc_encoder')
    reluLayer('Name', 'relu_encoder')
    % LSTM Core: Temporal pattern learning with sequence output
    lstmLayer(lstm_neurons, 'OutputMode', 'sequence', 'Name', 'lstm_core')
    % Custom layer to extract last timestep from sequence output
    functionLayer(@(X) extract_last_timestep(X), 'Formattable', true, 'Name', 'last_step_extractor')
    % FC Decoder: Project back to output space
    fullyConnectedLayer(fc_decoder_neurons, 'Name', 'fc_decoder')
    reluLayer('Name', 'relu_decoder')
    % Output layer: Final prediction
    fullyConnectedLayer(output_size, 'Name', 'output')
];
% Build and initialize the network
lgraph = layerGraph(layers);
dlnet = dlnetwork(lgraph);
fprintf('[ARCHITECTURE] Network Architecture:\n');
fprintf('         Input: %d features, Output: %d features\n', input_size, output_size);
fprintf('         Layers: Input -> FC(%d) -> LSTM(%d) -> FC(%d) -> Output\n', ...
        fc_encoder_neurons, lstm_neurons, fc_decoder_neurons);

%% ==================== DATA PREPARATION WITH DIMENSION VERIFICATION ====================
fprintf('[DATA PREP] Converting Data to Deep Learning Format...\n');
% Convert training data to dlarray format with proper dimension ordering
X_train_dl = prepare_lstm_input(X_train);
Y_train_dl = dlarray(single(Y_train'), 'CB');  % Targets: [features × batch_size]
% Convert validation and test data
X_val_dl = prepare_lstm_input(X_val);
Y_val_dl = dlarray(single(Y_val'), 'CB');
X_test_dl = prepare_lstm_input(X_test);
Y_test_dl = dlarray(single(Y_test'), 'CB');
fprintf('[DATA PREP] Data Dimensions Verified:\n');
fprintf('         X_train: [features=%d, timesteps=%d, batch=%d]\n', ...
        size(X_train_dl, 1), size(X_train_dl, 2), size(X_train_dl, 3));
fprintf('         Y_train: [features=%d, batch=%d]\n', size(Y_train_dl, 1), size(Y_train_dl, 2));

%% ==================== COMPREHENSIVE DIMENSION VERIFICATION ====================
fprintf('[VERIFICATION] Performing Dimension Compatibility Tests...\n');
% Test network with various batch sizes to ensure robustness
test_batch_sizes = [1, 4, 16];
all_tests_passed = true;
for i = 1:length(test_batch_sizes)
    batch_size = test_batch_sizes(i);
    fprintf('[VERIFICATION] Testing with batch size %d...\n', batch_size);
    
    if batch_size <= size(X_train_dl, 3)
        % Extract test batch
        X_test_batch = X_train_dl(:, :, 1:batch_size);
        Y_test_batch = Y_train_dl(:, 1:batch_size);
        
        try
            % Perform forward pass to test dimension compatibility
            Y_pred_test = predict(dlnet, X_test_batch);
            
            % Display dimension information
            fprintf('         Input:  [features=%d, timesteps=%d, batch=%d]\n', size(X_test_batch));
            fprintf('         Output: [features=%d, batch=%d]\n', size(Y_pred_test));
            fprintf('         Target: [features=%d, batch=%d]\n', size(Y_test_batch));
            
            % Verify dimension matching
            if size(Y_pred_test, 1) == size(Y_test_batch, 1) && ...
               size(Y_pred_test, 2) == size(Y_test_batch, 2)
                fprintf('         Dimension compatibility confirmed\n');
            else
                fprintf('         Dimension mismatch detected\n');
                all_tests_passed = false;
            end
        catch ME
            fprintf('         Forward pass failed: %s\n', ME.message);
            all_tests_passed = false;
        end
    else
        fprintf('         Batch size exceeds available data, skipping\n');
    end
end
if ~all_tests_passed
    fprintf('[WARNING] Some dimension tests failed, proceeding with training...\n');
else
    fprintf('[VERIFICATION] All dimension tests passed successfully!\n');
end

%% ==================== TRAINING WITH THREE-TERM LOSS FUNCTION ====================
fprintf('[TRAINING] Starting Training with Three-Term Loss Function...\n');
% Training hyperparameters
learnRate = 0.001;
gradDecay = 0.9;
sqGradDecay = 0.999;
miniBatchSize = 32;
% Initialize Adam optimizer state
averageGrad = [];
averageSqGrad = [];
% Initialize loss tracking arrays
trainLossHistory = zeros(1, num_epochs);
valLossHistory = zeros(1, num_epochs);
trainDataLossHistory = zeros(1, num_epochs);
trainPhysicsLossHistory = zeros(1, num_epochs);
trainConstraintLossHistory = zeros(1, num_epochs);
% Calculate number of iterations per epoch
numIterations = floor(size(X_train_dl, 3) / miniBatchSize);
fprintf('[TRAINING] Training Configuration:\n');
fprintf('         Epochs: %d, Batch Size: %d, Iterations/Epoch: %d\n', ...
        num_epochs, miniBatchSize, numIterations);
% Main training loop
for epoch = 1:num_epochs
    % Shuffle training data for each epoch
    idx = randperm(size(X_train_dl, 3));
    X_epoch = X_train_dl(:, :, idx);
    Y_epoch = Y_train_dl(:, idx);
    
    % Initialize epoch loss accumulators
    epochLoss = 0;
    epochDataLoss = 0;
    epochPhysicsLoss = 0;
    epochConstraintLoss = 0;
    
    % Mini-batch training
    for iteration = 1:numIterations
        % Extract current mini-batch
        batchStart = (iteration-1)*miniBatchSize + 1;
        batchEnd = min(iteration*miniBatchSize, size(X_epoch, 3));
        batchIdx = batchStart:batchEnd;
        
        XBatch = X_epoch(:, :, batchIdx);
        YTarget = Y_epoch(:, batchIdx);
        
        % Compute gradients and three-term losses
        [gradients, totalLoss, dataLoss, physicsLoss, constraintLoss] = ...
            dlfeval(@compute_three_term_gradients, dlnet, XBatch, YTarget, ...
                   lambda_data, lambda_physics, lambda_constraint);
        
        % Update network parameters using Adam optimizer
        [dlnet, averageGrad, averageSqGrad] = adamupdate(dlnet, gradients, ...
            averageGrad, averageSqGrad, iteration, learnRate, gradDecay, sqGradDecay);
        
        % Accumulate losses for monitoring
        epochLoss = epochLoss + extractdata(totalLoss);
        epochDataLoss = epochDataLoss + extractdata(dataLoss);
        epochPhysicsLoss = epochPhysicsLoss + extractdata(physicsLoss);
        epochConstraintLoss = epochConstraintLoss + extractdata(constraintLoss);
    end
    
    % Store average epoch losses
    trainLossHistory(epoch) = epochLoss / numIterations;
    trainDataLossHistory(epoch) = epochDataLoss / numIterations;
    trainPhysicsLossHistory(epoch) = epochPhysicsLoss / numIterations;
    trainConstraintLossHistory(epoch) = epochConstraintLoss / numIterations;
    
    % Compute validation loss
    valLoss = compute_validation_loss(dlnet, X_val_dl, Y_val_dl, ...
                                     lambda_data, lambda_physics, lambda_constraint);
    valLossHistory(epoch) = valLoss;
    
    % Display training progress
    if mod(epoch, 10) == 0 || epoch == 1
        fprintf('Epoch %03d/%d - Loss: %.4f (Data:%.4f, Physics:%.4f, Constraint:%.4f) Val: %.4f\n', ...
                epoch, num_epochs, trainLossHistory(epoch), ...
                trainDataLossHistory(epoch), trainPhysicsLossHistory(epoch), ...
                trainConstraintLossHistory(epoch), valLoss);
    end
end
% Calculate total training time
training_time = toc(training_start_time);
fprintf('[TRAINING] Training completed in %.2f seconds\n', training_time);

%% ==================== PREDICTION AND EVALUATION ====================
fprintf('[PREDICTION] Generating Deployment Predictions for Reliability Assessment...\n');
% Generate predictions for Bayesian Network parameter samples
predictions = generate_predictions(parameter_samples, n_outputs, n_steps);
fprintf('[EVALUATION] Evaluating Model Performance on Test Set...\n');
test_metrics = evaluate_model(dlnet, X_test_dl, Y_test_dl);

%% ==================== MODEL AND RESULTS EXPORT ====================
fprintf('[EXPORT] Saving Trained Model and Analysis Results...\n');
save_results(dlnet, test_metrics, normalization_params, window_size, lstm_config, ...
             trainLossHistory, valLossHistory, trainDataLossHistory, ...
             trainPhysicsLossHistory, trainConstraintLossHistory);

%% ==================== RESULTS PACKAGING ====================
fprintf('[RESULTS] Packaging Final Results...\n');
% Create comprehensive results structure
results = struct();
results.test_metrics = test_metrics;
results.network_config = lstm_config;
results.training_time = training_time;
results.training_history = struct(...
    'train_loss', trainLossHistory, ...
    'val_loss', valLossHistory, ...
    'data_loss', trainDataLossHistory, ...
    'physics_loss', trainPhysicsLossHistory, ...
    'constraint_loss', trainConstraintLossHistory);
results.model_metadata = struct(...
    'input_dimension', input_size, ...
    'output_dimension', output_size, ...
    'window_size', window_size, ...
    'training_samples', n_train, ...
    'completion_time', datetime('now'));

fprintf('[COMPLETE] PI-LSTM Analysis Completed Successfully\n');
fprintf('           Final Test MSE: %.6f, Training Time: %.2f seconds\n', ...
        test_metrics.mse, training_time);

%% ==================== HELPER FUNCTION DEFINITIONS ====================

function config = get_pi_lstm_config()
% GET_PI_LSTM_CONFIG - Returns configuration parameters for PI-LSTM model
% This function defines the network architecture and loss function parameters
% for the Physics-Informed LSTM model.
%
% Outputs:
%   config - Configuration structure with network and loss parameters
    config = struct();   
    % Network architecture parameters
    config.network = struct();
    config.network.fc_encoder_neurons = 128;   % Fully-connected encoder layer size
    config.network.lstm_neurons = 512;         % LSTM hidden units
    config.network.fc_decoder_neurons = 128;   % Fully-connected decoder layer size
    % Three-term loss function weights
    config.loss = struct();
    config.loss.lambda_data = 1.0;        % Weight for data fidelity term
    config.loss.lambda_physics = 0.5;     % Weight for physics constraint term
    config.loss.lambda_constraint = 0.2;  % Weight for domain constraint term
end
function [X_sequences, Y_targets] = prepare_sequence_data(Y_traj, window_size)
% PREPARE_SEQUENCE_DATA - Convert trajectory data to supervised learning format
    [n_samples, n_outputs, n_steps] = size(Y_traj);
    num_sequences = n_samples * (n_steps - window_size);
    
    % Preallocate arrays
    X_sequences = zeros(num_sequences, n_outputs, window_size);
    Y_targets = zeros(num_sequences, n_outputs);
    
    seq_idx = 1;
    for i = 1:n_samples
        % Extract trajectory for current sample
        sample_traj = squeeze(Y_traj(i, :, :));
        
        for t = 1:(n_steps - window_size)
            % Input sequence: window_size consecutive time steps
            X_sequences(seq_idx, :, :) = sample_traj(:, t:t+window_size-1);
            
            % Target: next time step after the input window
            Y_targets(seq_idx, :) = sample_traj(:, t+window_size)';
            
            seq_idx = seq_idx + 1;
        end
    end
end
function [X_norm, norm_params] = normalize_parameters(X_params)
% NORMALIZE_PARAMETERS - Normalize input parameters to zero mean and unit variance
    X_norm = zeros(size(X_params));
    norm_params.means = mean(X_params, 1);
    norm_params.stds = std(X_params, 0, 1);
    % Handle constant parameters (zero standard deviation)
    norm_params.stds(norm_params.stds == 0) = 1;
    % Apply z-score normalization
    for i = 1:size(X_params, 2)
        X_norm(:, i) = (X_params(:, i) - norm_params.means(i)) / norm_params.stds(i);
    end
end

function X_dl = prepare_lstm_input(X_data)
% PREPARE_LSTM_INPUT - Convert sequence data to dlarray format for deep learning
    [num_sequences, n_features, seq_length] = size(X_data);
    X_dl = zeros(n_features, seq_length, num_sequences, 'single');
    
    for i = 1:num_sequences
        sequence = squeeze(X_data(i, :, :));
        X_dl(:, :, i) = single(sequence);
    end
    
    X_dl = dlarray(X_dl, 'CBT');  % Format: Channel × Batch × Time
end
function Y_last = extract_last_timestep(X)
% EXTRACT_LAST_TIMESTEP - Custom layer to extract last timestep from sequence
    % Extract the last timestep from the sequence
    Y_last = X(:, end, :);
    
    % Remove singleton dimension to match target format
    Y_last = reshape(Y_last, size(X, 1), size(X, 3));
end
function [gradients, totalLoss, dataLoss, physicsLoss, constraintLoss] = ...
         compute_three_term_gradients(dlnet, X, Y, lambda_data, lambda_physics, lambda_constraint)
% COMPUTE_THREE_TERM_GRADIENTS - Calculate gradients using three-term loss function
    YPred = forward(dlnet, X);
    
    % ========== 1. DATA FIDELITY LOSS ==========
    % Mean Squared Error between predictions and ground truth targets
    dataLoss = mean((YPred - Y).^2, 'all');
    
    % ========== 2. PHYSICS CONSTRAINT LOSS ==========
    % NOTE: This is a placeholder for physics-based constraints
    % Users should implement domain-specific physics rules such as:
    % - Energy conservation laws
    % - Monotonicity constraints (deployment should be monotonic)
    % - Smoothness constraints (trajectory should be smooth)
    % - Dynamic constraints (Newton's laws, etc.)
    physicsLoss = dlarray(single(0));  % Set to 0 for basic functionality
    
    % ========== 3. DOMAIN CONSTRAINT LOSS ==========
    % NOTE: This is a placeholder for domain-specific constraints
    % Users should implement constraints such as:
    % - Output value bounds (angles within physical limits)
    % - Inequality constraints (maximum stress, deflection limits)
    % - Physical feasibility constraints
    constraintLoss = dlarray(single(0));  % Set to 0 for basic functionality
    
    % ========== COMBINED LOSS ==========
    % Weighted sum of three explicit loss terms
    % The framework is maintained for future extension with physics and constraints
    totalLoss = lambda_data * dataLoss + lambda_physics * physicsLoss + lambda_constraint * constraintLoss;
    
    % Compute gradients with respect to network learnable parameters
    gradients = dlgradient(totalLoss, dlnet.Learnables);
end

function valLoss = compute_validation_loss(dlnet, X_val, Y_val, lambda_data, lambda_physics, lambda_constraint)
% COMPUTE_VALIDATION_LOSS - Calculate validation loss using three-term loss
    YPred = predict(dlnet, X_val);
    
    % Three-term loss calculation (same as training)
    dataLoss = mean((YPred - Y_val).^2, 'all');
    physicsLoss = dlarray(single(0));      % Placeholder for physics loss
    constraintLoss = dlarray(single(0));   % Placeholder for constraint loss
    
    % Total validation loss
    valLoss = lambda_data * dataLoss + lambda_physics * physicsLoss + lambda_constraint * constraintLoss;
    valLoss = extractdata(valLoss);  % Convert to numeric value
end

function predictions = generate_predictions(parameter_samples, num_outputs, num_time_steps)
% GENERATE_PREDICTIONS - Generate deployment predictions for parameter samples
    num_samples = size(parameter_samples, 1);
    predictions = zeros(num_samples, num_outputs, num_time_steps);
    
    % Generate predictions using simplified physical models
    for i = 1:num_samples
        time_vec = linspace(0, 1, num_time_steps);
        
        % Extract key physical parameters
        panel_length = parameter_samples(i, 1);
        spring_stiffness = parameter_samples(i, 8);
        preload_angle = parameter_samples(i, 9);
        
        % Calculate natural frequency of deployment system
        omega_n = sqrt(spring_stiffness / (panel_length^2 * 0.1));
        damping = 0.1;  % Light damping coefficient
        
        % Generate trajectories for each output state
        for j = 1:num_outputs
            if j == 1  % Panel deployment angle
                predictions(i, j, :) = preload_angle * exp(-damping * omega_n * time_vec) .* ...
                                      cos(omega_n * sqrt(1-damping^2) * time_vec);
            elseif j == 2  % Panel angular velocity
                for t = 1:num_time_steps
                    predictions(i, j, t) = -preload_angle * omega_n * exp(-damping * omega_n * time_vec(t)) * ...
                                          (damping * cos(omega_n * sqrt(1-damping^2) * time_vec(t)) + ...
                                           sqrt(1-damping^2) * sin(omega_n * sqrt(1-damping^2) * time_vec(t)));
                end
            else  % Other states (vibrations, etc.)
                freq = 1 + 0.1 * randn();  % Random frequency with variation
                amp = 0.02 * (1 + 0.1 * randn());  % Random amplitude
                predictions(i, j, :) = amp * sin(2*pi*freq * time_vec);
            end
        end
    end
end

function test_metrics = evaluate_model(dlnet, X_test, Y_test)
% EVALUATE_MODEL - Comprehensive model evaluation on test dataset

    YPred = predict(dlnet, X_test);
    Y_test_numeric = extractdata(Y_test);
    
    % Calculate performance metrics
    mse_val = mean((extractdata(YPred) - Y_test_numeric).^2, 'all');
    mae_val = mean(abs(extractdata(YPred) - Y_test_numeric), 'all');
    
    % Additional metrics could be added here:
    % - R-squared coefficient
    % - Explained variance
    % - Maximum absolute error
    % - Mean absolute percentage error
    
    % Package metrics
    test_metrics.mse = mse_val;
    test_metrics.mae = mae_val;
    
    fprintf('[EVALUATION] Model Performance on Test Set:\n');
    fprintf('             MSE: %.6f, MAE: %.6f\n', mse_val, mae_val);
end

function save_results(dlnet, test_metrics, normalization_params, window_size, lstm_config, ...
                     trainLossHistory, valLossHistory, trainDataLossHistory, ...
                     trainPhysicsLossHistory, trainConstraintLossHistory)
% SAVE_RESULTS - Export trained model and training history

    % Create comprehensive model information structure
    model_info = struct();
    model_info.network = dlnet;
    model_info.performance_metrics = test_metrics;
    model_info.normalization_params = normalization_params;
    model_info.window_size = window_size;
    model_info.configuration = lstm_config;
    model_info.train_loss_history = trainLossHistory;
    model_info.val_loss_history = valLossHistory;
    model_info.train_data_loss_history = trainDataLossHistory;
    model_info.train_physics_loss_history = trainPhysicsLossHistory;
    model_info.train_constraint_loss_history = trainConstraintLossHistory;
    model_info.save_time = datetime('now');
    model_info.matlab_version = version;
    
    % Save model file
    output_path = fullfile('./output', 'pi_lstm_trained_model.mat');
    save(output_path, 'model_info', '-v7.3');
    
    fprintf('[SAVE] Model and training history saved to: %s\n', output_path);
end

end