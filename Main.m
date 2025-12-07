%% Main Function for Reliability Analysis of Satellite Solar Panel Deployment Systems
% Author: YD Fu
% Description: Integrated framework for satellite solar panel reliability analysis
%              combining Bayesian Networks, Physics-Informed LSTM, and reliability assessment
% License: Tongji University
% Dependencies: BNT, Deep Learning Toolbox, Statistics and Machine Learning Toolbox

clc; clear; close all;

%% ==================== MAIN CONFIGURATION ====================
fprintf('=== Solar Panel Deployment Reliability Analysis ===\n');
fprintf('Starting time: %s\n', datestr(now));
% Configuration structure for the entire pipeline
config = struct();
% Bayesian Network configuration for parameter correlation analysis
config.bn.data_path = './data/experimental_data.csv';  % Path to parameter data
config.bn.variable_names = {
    'panelLength', 'bodyLength', 'panelWidth', 'bodyWidth', ...
    'panelThickness', 'bodyHeight', 'gapSize', 'springStiffness', ...
    'preloadAngle', 'panelElasticity', 'panelDensity', 'bodyDensity'
};
config.bn.num_samples = 1000;  % Number of samples to generate
% PI-LSTM configuration  
config.lstm.data_path = './data/deployment_data.mat';  % Path to deployment data
config.lstm.num_epochs = 20;                           % Reduced for demo
config.lstm.window_size = 30;
config.lstm.train_ratio = 0.8;
config.lstm.val_ratio = 0.1;
config.lstm.test_ratio = 0.1;

% Reliability assessment configuration
config.reliability.num_time_points = 10;
config.reliability.max_eval_samples = 50;
config.reliability.beta_min = 0.5;
config.reliability.beta_max = 5.0;
config.reliability.tau = 2.0;

% Output configuration
config.output.save_results = true;
config.output.verbose = true;

fprintf('[MAIN] Configuration loaded successfully\n');

%% ==================== DATA GENERATION ====================
fprintf('\n=== STEP 0: Generating Demonstration Data ===\n');

% Generate BN training data
fprintf('[MAIN] Generating Bayesian Network training data...\n');
bn_training_data = generate_bn_training_data(config);
fprintf('[MAIN] Generated %d BN training samples\n', size(bn_training_data, 1));

% Generate PI-LSTM training data
fprintf('[MAIN] Generating PI-LSTM training data...\n');
[lstm_training_data, lstm_metadata] = generate_lstm_training_data(config);
fprintf('[MAIN] Generated PI-LSTM data: %d samples, %d time steps\n', ...
        size(lstm_training_data.X_params, 1), size(lstm_training_data.Y_trajectories, 3));

%% ==================== BAYESIAN NETWORK ANALYSIS ====================
fprintf('\n=== STEP 1: Bayesian Network Parameter Correlation Analysis ===\n');

try
    [bn_results, generated_samples] = run_bayesian_network_analysis(config, bn_training_data);
    fprintf('[MAIN] Bayesian Network analysis completed successfully\n');
    fprintf('       Generated %d parameter samples\n', size(generated_samples, 1));
catch ME
    fprintf('[ERROR] Bayesian Network analysis failed: %s\n', ME.message);
    fprintf('[MAIN] Using default parameter samples\n');
    % Generate fallback parameter samples
    generated_samples = generate_fallback_parameters(config);
end

%% ==================== PI-LSTM DEPLOYMENT PREDICTION ====================
fprintf('\n=== STEP 2: PI-LSTM Deployment Trajectory Prediction ===\n');

try
    [lstm_results, deployment_predictions] = run_pi_lstm_prediction(config, generated_samples, lstm_training_data);
    fprintf('[MAIN] PI-LSTM prediction completed successfully\n');
catch ME
    fprintf('[ERROR] PI-LSTM prediction failed: %s\n', ME.message);
    return;
end

%% ==================== RELIABILITY ASSESSMENT ====================
fprintf('\n=== STEP 3: Reliability Assessment ===\n');

try
    reliability_results = run_reliability_assessment(config, deployment_predictions, lstm_training_data);
    fprintf('[MAIN] Reliability assessment completed successfully\n');
catch ME
    fprintf('[ERROR] Reliability assessment failed: %s\n', ME.message);
    return;
end

%% ==================== RESULTS INTEGRATION AND ANALYSIS ====================
fprintf('\n=== STEP 4: Results Integration and Analysis ===\n');

% Combine all results
final_results = struct();
final_results.bayesian_network = bn_results;
final_results.lstm_prediction = lstm_results;
final_results.reliability = reliability_results;
final_results.timestamp = datetime('now');
final_results.configuration = config;

% Display summary
display_final_summary(final_results);

% Save complete results
if config.output.save_results
    save_final_results(final_results);
end

fprintf('\n=== Analysis Complete ===\n');
fprintf('Total time: %s\n', datestr(now));
fprintf('Results saved to: ./output/final_reliability_analysis.mat\n');

%% ==================== DATA GENERATION FUNCTIONS ====================

function bn_data = generate_bn_training_data(config)
% GENERATE_BN_TRAINING_DATA - Generate synthetic data for BN training
    num_samples = config.bn.num_samples;
    num_params = length(config.bn.variable_names);
    
    rng(42); % For reproducibility
    
    % Generate base samples with correlations
    base_samples = randn(num_samples, num_params);
    
    % Create correlation structure
    corr_matrix = eye(num_params);
    % Correlate geometry parameters
    corr_matrix(1:3, 1:3) = 0.7;
    corr_matrix(4:6, 4:6) = 0.6;
    corr_matrix(7:9, 7:9) = 0.5;
    corr_matrix(10:12, 10:12) = 0.4;
    
    [U,S,~] = svd(corr_matrix);
    corr_transform = U * sqrt(S);
    correlated_samples = base_samples * corr_transform';
    
    % Scale to reasonable parameter ranges
    param_ranges = [
        0.5, 2.0;   % panelLength [m]
        1.0, 3.0;   % bodyLength [m]
        0.3, 1.0;   % panelWidth [m]
        0.5, 1.5;   % bodyWidth [m]
        0.01, 0.05; % panelThickness [m]
        0.1, 0.5;   % bodyHeight [m]
        0.001, 0.01; % gapSize [m]
        10, 100;    % springStiffness [N¡¤m/rad]
        0.1, 1.0;   % preloadAngle [rad]
        1e9, 1e10;  % panelElasticity [Pa]
        1000, 3000; % panelDensity [kg/m?]
        2000, 5000  % bodyDensity [kg/m?]
    ];
    
    bn_data = zeros(size(correlated_samples));
    for i = 1:num_params
        % Normalize to [0,1] then scale to parameter range
        col = correlated_samples(:, i);
        col_normalized = (col - min(col)) / (max(col) - min(col));
        bn_data(:, i) = param_ranges(i,1) + col_normalized * (param_ranges(i,2) - param_ranges(i,1));
    end
    
    fprintf('[DATA] Generated %d BN training samples\n', num_samples);
end

function [training_data, metadata] = generate_lstm_training_data(config)
% GENERATE_LSTM_TRAINING_DATA - Generate synthetic deployment trajectory data
    num_samples = 50;  % Reduced for demo
    num_time_steps = 100;
    num_params = 12;
    num_outputs = 12;   % 12 state functions
    
    rng(123); % For reproducibility
    
    % Generate parameter samples (similar to BN but different)
    X_params = randn(num_samples, num_params);
    
    % Create parameter correlations
    param_corr = eye(num_params);
    param_corr(1:6, 1:6) = 0.6;
    [U,S,~] = svd(param_corr);
    corr_transform = U * sqrt(S);
    X_params = X_params * corr_transform';
    
    % Scale parameters
    param_scales = [2.0, 3.0, 1.0, 1.5, 0.05, 0.5, 0.01, 100, 1.0, 1e10, 3000, 5000];
    for i = 1:num_params
        X_params(:, i) = abs(X_params(:, i)) * param_scales(i);
    end
    
    % Generate time vector
    time_vec = linspace(0, 1, num_time_steps);
    
    % Generate deployment trajectories (simplified physics)
    Y_trajectories = zeros(num_samples, num_outputs, num_time_steps);
    
    for i = 1:num_samples
        % Extract key parameters for this sample
        panel_length = X_params(i, 1);
        spring_stiffness = X_params(i, 8);
        preload_angle = X_params(i, 9);
        
        % Generate realistic deployment trajectories
        for j = 1:num_outputs
            if j <= 3  % Panel states (angle, angular velocity, angular acceleration)
                % Panel deployment dynamics
                omega_n = sqrt(spring_stiffness / (panel_length^2 * 0.1)); % natural frequency
                damping = 0.1;
                
                if j == 1 % Angle
                    Y_trajectories(i, j, :) = preload_angle * exp(-damping * omega_n * time_vec) .* ...
                                            cos(omega_n * sqrt(1-damping^2) * time_vec);
                elseif j == 2 % Angular velocity
                    for t = 1:num_time_steps
                        Y_trajectories(i, j, t) = -preload_angle * omega_n * exp(-damping * omega_n * time_vec(t)) * ...
                                                (damping * cos(omega_n * sqrt(1-damping^2) * time_vec(t)) + ...
                                                 sqrt(1-damping^2) * sin(omega_n * sqrt(1-damping^2) * time_vec(t)));
                    end
                else % Angular acceleration
                    for t = 1:num_time_steps
                        Y_trajectories(i, j, t) = preload_angle * omega_n^2 * exp(-damping * omega_n * time_vec(t)) * ...
                                                ((2*damping^2 - 1) * cos(omega_n * sqrt(1-damping^2) * time_vec(t)) + ...
                                                 2*damping*sqrt(1-damping^2) * sin(omega_n * sqrt(1-damping^2) * time_vec(t)));
                    end
                end
                
            elseif j <= 9 % Body states (displacement, velocity, acceleration in X,Y)
                % Body vibration responses
                body_freq = 2.0 + 0.5 * randn();
                body_amp = 0.01 * (1 + 0.2 * randn());
                
                if j <= 6 % Displacements
                    phase_shift = (j-4) * pi/6;
                    Y_trajectories(i, j, :) = body_amp * sin(2*pi*body_freq * time_vec + phase_shift);
                else % Velocities
                    phase_shift = (j-7) * pi/6;
                    Y_trajectories(i, j, :) = 2*pi*body_freq * body_amp * cos(2*pi*body_freq * time_vec + phase_shift);
                end
                
            else % Body states (theta, omega, alpha)
                % Body rotational dynamics
                body_rot_freq = 1.0 + 0.2 * randn();
                body_rot_amp = 0.05 * (1 + 0.1 * randn());
                
                if j == 10 % Theta
                    Y_trajectories(i, j, :) = body_rot_amp * sin(2*pi*body_rot_freq * time_vec);
                elseif j == 11 % Omega
                    Y_trajectories(i, j, :) = 2*pi*body_rot_freq * body_rot_amp * cos(2*pi*body_rot_freq * time_vec);
                else % Alpha
                    Y_trajectories(i, j, :) = -(2*pi*body_rot_freq)^2 * body_rot_amp * sin(2*pi*body_rot_freq * time_vec);
                end
            end
        end
        
        % Add some noise
        Y_trajectories(i, :, :) = Y_trajectories(i, :, :) + 0.01 * randn(size(Y_trajectories(i, :, :)));
        
        if mod(i, 100) == 0
            fprintf('[DATA] Generated trajectory %d/%d\n', i, num_samples);
        end
    end
    
    % Package data
    training_data.X_params = X_params;
    training_data.Y_trajectories = Y_trajectories;
    training_data.time_vec = time_vec;
    
    metadata.num_samples = num_samples;
    metadata.num_time_steps = num_time_steps;
    metadata.num_params = num_params;
    metadata.num_outputs = num_outputs;
    metadata.generation_time = datetime('now');
    
    fprintf('[DATA] Generated LSTM training data: %d samples, %d outputs, %d time steps\n', ...
            num_samples, num_outputs, num_time_steps);
end

%% ==================== HELPER FUNCTIONS ====================

function display_final_summary(results)
% DISPLAY_FINAL_SUMMARY - Display comprehensive analysis summary
    fprintf('\n=== FINAL ANALYSIS SUMMARY ===\n');
    
    % Bayesian Network summary
    if isfield(results.bayesian_network, 'bic_scores')
        bic_prior = results.bayesian_network.bic_scores(1);
        bic_learned = results.bayesian_network.bic_scores(2);
        fprintf('Bayesian Network:\n');
        fprintf('  BIC Score - Prior: %.2f, Learned: %.2f\n', bic_prior, bic_learned);
        fprintf('  BIC Improvement: %.2f\n', bic_learned - bic_prior);
    end
    
    % PI-LSTM summary
    if isfield(results.lstm_prediction, 'test_metrics')
        fprintf('PI-LSTM Prediction:\n');
        fprintf('  Test MSE: %.6f\n', results.lstm_prediction.test_metrics.mse);
    end
    
    % Reliability summary
    if isfield(results.reliability, 'overall_reliability_real')
        fprintf('Reliability Assessment:\n');
        fprintf('  Overall Reliability: %.4f (%.2f%%)\n', ...
            results.reliability.overall_reliability_real, ...
            results.reliability.overall_reliability_real * 100);
    end
    
    fprintf('\n');
end

function save_final_results(results)
% SAVE_FINAL_RESULTS - Save complete analysis results
    output_dir = './output';
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    save(fullfile(output_dir, 'final_reliability_analysis.mat'), 'results', '-v7.3');
    fprintf('[SAVE] Complete results saved to: %s\n', ...
        fullfile(output_dir, 'final_reliability_analysis.mat'));
end

function fallback_samples = generate_fallback_parameters(config)
% GENERATE_FALLBACK_PARAMETERS - Generate fallback parameter samples if BN fails
    num_params = length(config.bn.variable_names);
    num_samples = config.bn.num_samples;
    
    % Generate random samples with reasonable correlations
    rng(42); % For reproducibility
    base_samples = randn(num_samples, num_params);
    
    % Add some correlation structure
    corr_matrix = eye(num_params);
    corr_matrix(1:3, 1:3) = 0.7; % Correlate geometry parameters
    corr_matrix(4:6, 4:6) = 0.6; % Correlate body parameters
    
    [U,S,~] = svd(corr_matrix);
    corr_transform = U * sqrt(S);
    fallback_samples = base_samples * corr_transform';
    
    % Scale to reasonable ranges
    param_ranges = [
        0.5, 2.0;   % panelLength [m]
        1.0, 3.0;   % bodyLength [m]
        0.3, 1.0;   % panelWidth [m]
        0.5, 1.5;   % bodyWidth [m]
        0.01, 0.05; % panelThickness [m]
        0.1, 0.5;   % bodyHeight [m]
        0.001, 0.01; % gapSize [m]
        10, 100;    % springStiffness [N¡¤m/rad]
        0.1, 1.0;   % preloadAngle [rad]
        1e9, 1e10;  % panelElasticity [Pa]
        1000, 3000; % panelDensity [kg/m?]
        2000, 5000  % bodyDensity [kg/m?]
    ];
    
    for i = 1:num_params
        fallback_samples(:,i) = param_ranges(i,1) + ...
                               (param_ranges(i,2) - param_ranges(i,1)) * ...
                               (fallback_samples(:,i) - min(fallback_samples(:,i))) / ...
                               (max(fallback_samples(:,i)) - min(fallback_samples(:,i)));
    end
    
    fprintf('[FALLBACK] Generated %d fallback parameter samples\n', num_samples);
end
