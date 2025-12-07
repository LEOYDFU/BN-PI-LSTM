function results = run_reliability_assessment(~, deployment_predictions, lstm_training_data)
%% Reliability Assessment code
% Author: YD Fu
% Description: Evaluation of Time-Varying and Instantaneous Reliability of Satellite Systems
% License: Tongji University

%% ==================== ENVIRONMENT SETUP ====================
fprintf('[SETUP] Initializing reliability assessment system...\n');

% Load parameters from configuration
reliability_config = get_reliability_config();

% Extract parameters from configuration
beta_min = reliability_config.weights.beta_min;
beta_max = reliability_config.weights.beta_max;
tau = reliability_config.weights.tau;
eps = reliability_config.system.eps;
num_time_points = reliability_config.system.num_time_points;
t0 = reliability_config.system.t0;
tM = reliability_config.system.tM;
max_eval_samples = reliability_config.system.max_eval_samples;

% Generate time vector
time_vec = linspace(t0, tM, num_time_points);

fprintf('[SETUP] Reliability assessment configuration loaded\n');

%% ==================== DATA PREPARATION ====================
fprintf('[DATA] Preparing reliability assessment data...\n');

% Extract data from LSTM training data for reference
Y_full = lstm_training_data.Y_trajectories;  % Real deployment data
num_real_samples = size(Y_full, 1);
num_time_steps = size(Y_full, 3);
num_outputs = size(Y_full, 2);

% Use predicted data from PI-LSTM
model_output_pred = deployment_predictions;  % Predicted deployment data
num_pred_samples = size(model_output_pred, 1);

% Select evaluation samples
eval_samples = min(max_eval_samples, num_pred_samples);
selected_indices = 1:eval_samples;

fprintf('[DATA] Using %d samples for reliability assessment\n', eval_samples);

%% ==================== STATE FUNCTION DEFINITION ====================
fprintf('[STATES] Defining state functions...\n');

% State function definition
state_functions = {
    'Panel-Angle', 'Panel-AngularVelocity', 'Panel-AngularAcceleration', ...
    'Body-DispX', 'Body-DispY', 'Body-VelX', 'Body-VelY', ...
    'Body-AccX', 'Body-AccY', 'Body-Theta', 'Body-Omega', 'Body-Alpha'};

num_states = length(state_functions);

% Validate state function count
if num_states ~= num_outputs
    fprintf('[STATES] Adjusting state functions to match output dimension: %d\n', num_outputs);
    if num_outputs > num_states
        % Add generic state functions
        for i = num_states+1:num_outputs
            state_functions{i} = sprintf('State_%02d', i);
        end
    else
        % Truncate state functions
        state_functions = state_functions(1:num_outputs);
    end
    num_states = num_outputs;
end

fprintf('[STATES] Defined %d state functions\n', num_states);

%% ==================== DATA INTERPOLATION ====================
fprintf('[PREPARE] Preparing reliability assessment data...\n');

% Initialize data storage
true_output = zeros(num_states, num_time_points);
model_output_real = zeros(eval_samples, num_states, num_time_points);
model_output_pred_interp = zeros(eval_samples, num_states, num_time_points);
error_limit = zeros(num_states, num_time_points);

% Time interpolation setup
interp_time = time_vec;
main_time = linspace(0, 1, num_time_steps);

% For demonstration, we'll use a subset of real data as reference
% and the predicted data from PI-LSTM
for i = 1:eval_samples
    idx = selected_indices(i);
    
    % Get real output (reference data)
    if idx <= num_real_samples
        true_output_full = squeeze(Y_full(idx, :, :));
    else
        % If not enough real samples, use first sample
        true_output_full = squeeze(Y_full(1, :, :));
    end
    
    % Get predicted output
    pred_output_full = squeeze(model_output_pred(idx, :, :));
    
    % Interpolate each state function to reliability assessment time grid
    for k = 1:num_states
        if k <= size(true_output_full, 1) && k <= size(pred_output_full, 1)
            % Real data interpolation
            if size(true_output_full, 2) == num_time_steps
                real_trajectory = true_output_full(k, :);
            else
                real_trajectory = true_output_full(:, k)';
            end
            
            % Predicted data interpolation  
            if size(pred_output_full, 2) == num_time_steps
                pred_trajectory = pred_output_full(k, :);
            else
                pred_trajectory = pred_output_full(:, k)';
            end
            
            model_output_real(i, k, :) = interp1(main_time, real_trajectory, interp_time, 'linear', 'extrap');
            model_output_pred_interp(i, k, :) = interp1(main_time, pred_trajectory, interp_time, 'linear', 'extrap');
        end
    end
    
    if mod(i, 10) == 0
        fprintf('[PREPARE] Completed data interpolation for %d/%d samples\n', i, eval_samples);
    end
end

% Compute theoretical output (reference values) as weighted average
fprintf('[PREPARE] Computing theoretical output...\n');
for k = 1:num_states
    real_mean = squeeze(mean(model_output_real(:, k, :), 1));
    pred_mean = squeeze(mean(model_output_pred_interp(:, k, :), 1));
    true_output(k, :) = 0.7 * real_mean + 0.3 * pred_mean;  % Weighted combination
end

% Set error limits based on state function characteristics
fprintf('[PREPARE] Setting error limits...\n');
for k = 1:num_states
    all_vals_real = squeeze(model_output_real(:, k, :));
    all_vals_pred = squeeze(model_output_pred_interp(:, k, :));
    all_vals = [all_vals_real(:); all_vals_pred(:)];
    state_std = std(all_vals);
    
    % Set error coefficients based on state function type
    if contains(state_functions{k}, 'Acc') || contains(state_functions{k}, 'Alpha')
        error_coeff = 2.0;
        error_type = 'Acceleration';
    elseif contains(state_functions{k}, 'Vel') || contains(state_functions{k}, 'Omega')
        error_coeff = 1.5;
        error_type = 'Velocity';
    else
        error_coeff = 1.0;
        error_type = 'Displacement/Angle';
    end
    
    base_error = error_coeff * state_std;
    error_limit(k, :) = base_error;
    
    if mod(k, 4) == 0
        fprintf('[PREPARE] State %d (%s): Type=%s, Std=%.4f, BaseError=%.4f\n', ...
            k, state_functions{k}, error_type, state_std, base_error);
    end
end

%% ==================== RELIABILITY ASSESSMENT ====================
% Reliability assessment based on predicted data
fprintf('\n=== RELIABILITY ASSESSMENT ===\n');
[reliability_results, overall_reliability] = compute_reliability_analysis(...
    model_output_pred_interp, true_output, error_limit, time_vec, ...
    beta_min, beta_max, tau, eps, state_functions, 'Predicted Data');

%% ==================== RESULTS EXPORT ====================
fprintf('[EXPORT] Saving reliability assessment results...\n');

% Create results structure
results = struct();
results.reliability = reliability_results;
results.overall_reliability_real = overall_reliability;  % Using predicted as real for demo
results.overall_reliability_pred = overall_reliability;
results.time_vec = time_vec;
results.state_functions = state_functions;
results.eval_samples = eval_samples;
results.model_output_pred = model_output_pred_interp;
results.true_output = true_output;
results.error_limit = error_limit;
results.configuration = reliability_config;

% Save results
output_dir = './output';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
save(fullfile(output_dir, 'reliability_assessment_results.mat'), 'results', '-v7.3');

% Display summary
fprintf('\n=== RELIABILITY ASSESSMENT SUMMARY ===\n');
fprintf('Overall Reliability: %.4f (%.2f%%)\n', overall_reliability, overall_reliability*100);
fprintf('Assessment Samples: %d\n', eval_samples);
fprintf('Time Points: %d\n', num_time_points);
fprintf('State Functions: %d\n', num_states);
fprintf('Results saved to: ./output/reliability_assessment_results.mat\n');

fprintf('\nReliability assessment completed!\n');

%% ==================== HELPER FUNCTION DEFINITIONS ====================

function config = get_reliability_config()
% GET_RELIABILITY_CONFIG - Configuration function for reliability assessment parameters
    
    config = struct();
    
    % Adaptive weight parameters
    config.weights = struct();
    config.weights.beta_min = 0.5;     % Minimum beta value
    config.weights.beta_max = 5;       % Maximum beta value
    config.weights.tau = 2;            % Decay coefficient
    
    % System parameters
    config.system = struct();
    config.system.num_time_points = 100; % Number of time discretization points
    config.system.t0 = 0;             % Start time
    config.system.tM = 1;             % End time
    config.system.eps = 1e-6;         % Numerical stability term
    config.system.max_eval_samples = 50; % Maximum samples for evaluation
    
    fprintf('[CONFIG] Reliability assessment configuration loaded\n');
end

function [reliability, overall_reliability] = compute_reliability_analysis(...
    model_output, true_output, error_limit, time_vec, ...
    beta_min, beta_max, tau, eps, state_functions, data_type)

    num_samples = size(model_output, 1);
    num_states = size(model_output, 2);
    num_time_points = length(time_vec);
    
    fprintf('[RELIABILITY] Computing error functions and limit state functions for %s...\n', data_type);
    
    % Initialize error matrix E and G matrix
    E_matrix = zeros(num_samples, num_states, num_time_points);
    G_matrix = zeros(num_samples, num_states, num_time_points);
    
    for i = 1:num_samples
        for t = 1:num_time_points
            % Compute error function E = ¦È - O (model output - true output)
            E_matrix(i, :, t) = squeeze(model_output(i, :, t)) - true_output(:, t)';
            
            % Compute limit state function components G* = e(t) - |E|
            G_matrix(i, :, t) = error_limit(:, t)' - abs(squeeze(E_matrix(i, :, t)));
        end
        
        if mod(i, 10) == 0
            fprintf('[RELIABILITY] Completed error calculation for %d/%d samples\n', i, num_samples);
        end
    end
    
    fprintf('[RELIABILITY] Computing reliability for %s...\n', data_type);
    
    % Initialize reliability matrices
    reliability_min = ones(num_samples, num_time_points);
    reliability_actual = ones(num_samples, num_time_points);
    reliability_max = ones(num_samples, num_time_points);
    g_values = zeros(num_samples, num_time_points);
    
    for i = 1:num_samples
        % Extract all state functions for current sample
        G_vectors = zeros(num_states, num_time_points);
        for k = 1:num_states
            G_vectors(k, :) = squeeze(G_matrix(i, k, :))';
        end
        
        % Calculate mean absolute values for each state function
        m_vals = mean(abs(G_vectors), 2);
        
        % Find state functions with minimum and maximum means
        [~, idx_min] = min(m_vals);
        [~, idx_max] = max(m_vals);
        
        gmin_vector = G_vectors(idx_min, :);
        gmax_vector = G_vectors(idx_max, :);
        
        % Calculate reliability bounds using gmin and gmax vectors
        reliability_min(i, :) = (gmin_vector >= 0);
        reliability_max(i, :) = (gmax_vector >= 0);
        
        % Calculate actual reliability (adaptive weight method)
        for t = 1:num_time_points
            G_t = squeeze(G_matrix(i, :, t));
            
            % Calculate adaptive weights
            mu = mean(G_t);
            sigma = std(G_t) + eps;
            kappa = sigma / max(abs(mu), eps);
            beta = beta_min + (beta_max - beta_min) * exp(-tau * kappa);
            d = abs(G_t - mu) / sigma;
            w = exp(-beta * d);
            alpha = w / sum(w);
            
            % Calculate weighted limit state function
            g_actual = dot(alpha, G_t);
            g_values(i, t) = g_actual;
            reliability_actual(i, t) = (g_actual >= 0);
        end
        
        if mod(i, 10) == 0
            fprintf('[RELIABILITY] Completed reliability calculation for %d/%d samples\n', i, num_samples);
        end
    end
    
    % Calculate time-varying and instantaneous reliability
    time_varying_reliability_min = mean(reliability_min, 2);
    time_varying_reliability_actual = mean(reliability_actual, 2);
    time_varying_reliability_max = mean(reliability_max, 2);
    
    instant_reliability_min = mean(reliability_min, 1);
    instant_reliability_actual = mean(reliability_actual, 1);
    instant_reliability_max = mean(reliability_max, 1);
    
    % Calculate overall reliability
    overall_reliability_min = mean(time_varying_reliability_min);
    overall_reliability_actual = mean(time_varying_reliability_actual);
    overall_reliability_max = mean(time_varying_reliability_max);
    
    % Save results to structure
    reliability = struct();
    reliability.E_matrix = E_matrix;
    reliability.G_matrix = G_matrix;
    reliability.reliability_min = reliability_min;
    reliability.reliability_actual = reliability_actual;
    reliability.reliability_max = reliability_max;
    reliability.g_values = g_values;
    reliability.time_varying_reliability_min = time_varying_reliability_min;
    reliability.time_varying_reliability_actual = time_varying_reliability_actual;
    reliability.time_varying_reliability_max = time_varying_reliability_max;
    reliability.instant_reliability_min = instant_reliability_min;
    reliability.instant_reliability_actual = instant_reliability_actual;
    reliability.instant_reliability_max = instant_reliability_max;
    reliability.overall_reliability_min = overall_reliability_min;
    reliability.overall_reliability_actual = overall_reliability_actual;
    reliability.overall_reliability_max = overall_reliability_max;
    reliability.data_type = data_type;
    
    fprintf('[RELIABILITY] %s Results:\n', data_type);
    fprintf('  g_min Reliability: %.4f (%.2f%%)\n', overall_reliability_min, overall_reliability_min*100);
    fprintf('  Actual Reliability: %.4f (%.2f%%)\n', overall_reliability_actual, overall_reliability_actual*100);
    fprintf('  g_max Reliability: %.4f (%.2f%%)\n', overall_reliability_max, overall_reliability_max*100);
    
    % Ensure output parameter is correctly assigned
    overall_reliability = overall_reliability_actual;
end
end