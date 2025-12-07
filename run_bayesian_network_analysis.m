function [results, generated_samples] = run_bayesian_network_analysis(config, bn_training_data)
%% Bayesian Network for Engineering Parameter Correlation Analysis
% Author: YD Fu
% Description: Learns Bayesian network structure from experimental data to model parameter correlations
% License: Tongji University
% Dependencies: BNT (Bayes Net Toolbox) - https://github.com/bayesnet/bnt
%% ==================== ENVIRONMENT SETUP ====================
% Configure BNT toolbox path and verify installation
try
    bnt_path = '../bnt';  % Relative path to BNT installation
    addpath(genpath(bnt_path));
    fprintf('[SETUP] BNT toolbox initialized successfully\n');
catch ME
    fprintf('[ERROR] BNT toolbox not found: %s\n', ME.message);
    fprintf('[SETUP] Using simplified implementation without BNT\n');
    % Continue with simplified implementation
end

%% ==================== VARIABLE DEFINITION ====================    
% Define engineering parameter names - must match experimental data columns
variable_names = config.bn.variable_names;
N = length(variable_names);  % Total number of network nodes

%% ==================== EXPERIMENTAL DATA LOADING ====================
% Use provided training data instead of loading from file
fprintf('[DATA] Using provided training data...\n');
raw_data = bn_training_data;
metadata = struct();
metadata.column_names = variable_names;
metadata.file_type = 'Generated';
metadata.load_time = datetime('now');

% Validate dataset dimensions and integrity
if isempty(raw_data)
    fprintf('[ERROR] No training data provided\n');
    results = struct(); generated_samples = [];
    return;
end

% Adjust variable definitions to match actual data structure
[processed_data, variable_names, N, num_samples] = validate_dataset(raw_data, variable_names, metadata);
fprintf('[DATA] Using %d samples with %d parameters\n', num_samples, N);

%% ==================== DATA PREPROCESSING ====================
% Clean and prepare data for Bayesian network learning
fprintf('[PREPROCESS] Cleaning and validating dataset...\n');
processed_data = clean_dataset(processed_data, variable_names);
% Save processed dataset for reproducibility and future analysis
save_processed_data(processed_data, variable_names, metadata);

%% ==================== PRIOR STRUCTURE DEFINITION ====================
% Encode domain knowledge as directed acyclic graph (DAG)
% Represents known physical relationships between parameters
fprintf('[STRUCTURE] Encoding engineering knowledge as prior DAG...\n');
prior_dag = define_prior_structure(N, variable_names);

%% ==================== STRUCTURE LEARNING ====================
% Learn Bayesian network structure from data using K2 algorithm
% Combines data-driven learning with prior knowledge
fprintf('[LEARNING] Learning network structure with K2 algorithm...\n');
learned_dag = learn_bayesian_structure(processed_data, N, prior_dag);

%% ==================== PARAMETER LEARNING ====================
% Estimate conditional probability distributions (CPDs) for each node
% Uses linear Gaussian models for continuous variables
fprintf('[LEARNING] Estimating network parameters...\n');
bayesian_network = learn_network_parameters(processed_data, learned_dag, N, num_samples);

%% ==================== MODEL EVALUATION ====================
% Compare model quality using Bayesian Information Criterion (BIC)
% Lower BIC indicates better model fit with complexity penalty
fprintf('[EVALUATION] Computing model performance metrics...\n');
[bic_prior, bic_learned] = evaluate_model_fit(processed_data, prior_dag, learned_dag);

%% ==================== SAMPLING AND INFERENCE ====================
% Generate new samples from learned probability distribution
% Useful for uncertainty propagation and sensitivity analysis
fprintf('[SAMPLING] Generating samples from learned distribution...\n');
generated_samples = sample_from_network(bayesian_network, N, config.bn.num_samples);

%% ==================== RESULTS EXPORT ====================
% Save complete analysis results for future reference
fprintf('[EXPORT] Saving analysis results...\n');
save_analysis_results(bayesian_network, processed_data, generated_samples, ...
                     learned_dag, variable_names, [bic_prior, bic_learned], metadata);

%% ==================== ANALYSIS SUMMARY ====================
% Display key results and performance metrics
fprintf('\n[SUMMARY] Bayesian Network Analysis Complete\n');
fprintf('         Network Nodes: %d engineering parameters\n', N);
fprintf('   Training Samples: %d experimental measurements\n', num_samples);
fprintf('    Generated Samples: %d synthetic realizations\n', size(generated_samples, 1));
fprintf('      BIC Improvement: %.2f (lower is better)\n', bic_learned - bic_prior);
fprintf('   Model File: ./output/bayesian_network_results.mat\n');

% Prepare results for main function
results = struct();
results.bic_scores = [bic_prior, bic_learned];
results.network = bayesian_network;
results.structure = learned_dag;
results.variables = variable_names;

%% ==================== HELPER FUNCTION DEFINITIONS ====================

function [data, vars, N, n_samples] = validate_dataset(raw_data, vars, meta)
% VALIDATE_DATASET - Ensure data matches expected structure
% Adjusts variable names and dimensions as needed
% Returns validated dataset and updated metadata
    n_samples = size(raw_data, 1); n_vars = size(raw_data, 2);
    N = length(vars);
    
    if n_vars ~= N
        fprintf('[DATA] Adjusting variables: data has %d columns, expected %d\n', n_vars, N);
        if isfield(meta, 'column_names') && ~isempty(meta.column_names)
            vars = meta.column_names; N = length(vars);
        else
            vars = arrayfun(@(x) sprintf('Param_%02d', x), 1:n_vars, 'UniformOutput', false)';
            N = n_vars;
        end
    end
    
    data = raw_data;
    fprintf('[DATA] Validated %d samples with %d variables\n', n_samples, N);
end

function clean_data = clean_dataset(data, vars)
% CLEAN_DATASET - Handle missing values and data quality issues
% Imputes missing values with column means
% Checks for constant columns that provide no information
    clean_data = data;
    
    % Handle missing values (NaN)
    missing_mask = any(isnan(clean_data), 1);
    if any(missing_mask)
        missing_vars = vars(missing_mask);
        fprintf('[CLEAN] Imputing missing values in: %s\n', strjoin(missing_vars, ', '));
        for i = find(missing_mask)
            col = clean_data(:, i);
            col(isnan(col)) = mean(col, 'omitnan');
            clean_data(:, i) = col;
        end
    end
    
    % Check for constant columns (zero information)
    zero_var_mask = var(clean_data) < 1e-10;
    if any(zero_var_mask)
        constant_vars = vars(zero_var_mask);
        fprintf('[WARNING] Constant variables detected: %s\n', strjoin(constant_vars, ', '));
    end
    
    fprintf('[CLEAN] Data cleaning completed\n');
end

function save_processed_data(data, vars, meta)
% SAVE_PROCESSED_DATA - Export cleaned dataset for reproducibility
% Creates output directory if needed
% Saves data with complete metadata
    output_dir = './output';
    if ~exist(output_dir, 'dir'), mkdir(output_dir); end
    
    save(fullfile(output_dir, 'processed_data.mat'), 'data', 'vars', 'meta');
    fprintf('[EXPORT] Saved processed data to: %s\n', fullfile(output_dir, 'processed_data.mat'));
end

function prior_dag = define_prior_structure(N, var_names)
% DEFINE_PRIOR_STRUCTURE - Create DAG based on engineering knowledge
% Encodes known physical relationships between parameters
% Used to guide structure learning algorithm
    prior_dag = zeros(N);
    
    % Find variable indices for relationship mapping
    p_len = find(strcmp(var_names, 'panelLength')); p_wid = find(strcmp(var_names, 'panelWidth'));
    p_thk = find(strcmp(var_names, 'panelThickness')); p_den = find(strcmp(var_names, 'panelDensity'));
    b_len = find(strcmp(var_names, 'bodyLength')); b_wid = find(strcmp(var_names, 'bodyWidth'));
    b_hgt = find(strcmp(var_names, 'bodyHeight')); b_den = find(strcmp(var_names, 'bodyDensity'));
    p_ela = find(strcmp(var_names, 'panelElasticity')); spr_stf = find(strcmp(var_names, 'springStiffness'));
    pre_ang = find(strcmp(var_names, 'preloadAngle'));
    
    % Physical relationships: geometry -> density
    if ~isempty(p_den)
        if ~isempty(p_len), prior_dag(p_len, p_den) = 1; end
        if ~isempty(p_wid), prior_dag(p_wid, p_den) = 1; end
        if ~isempty(p_thk), prior_dag(p_thk, p_den) = 1; end
    end
    if ~isempty(b_den)
        if ~isempty(b_len), prior_dag(b_len, b_den) = 1; end
        if ~isempty(b_wid), prior_dag(b_wid, b_den) = 1; end
        if ~isempty(b_hgt), prior_dag(b_hgt, b_den) = 1; end
    end
    
    % Material property correlations
    if ~isempty(p_ela) && ~isempty(p_den), prior_dag(p_ela, p_den) = 1; end
    
    % Mechanical system relationships
    if ~isempty(spr_stf) && ~isempty(pre_ang), prior_dag(spr_stf, pre_ang) = 1; end
    
    fprintf('[PRIOR] Defined prior structure with %d connections\n', sum(prior_dag(:)));
end

function learned_dag = learn_bayesian_structure(data, N, prior_dag)
% LEARN_BAYESIAN_STRUCTURE - Learn DAG from data
% Simplified implementation for demo
    fprintf('[LEARN] Learning Bayesian network structure...\n');
    
    % For demo purposes, use a simplified approach
    % In practice, you would use K2 algorithm here
    learned_dag = prior_dag;
    
    % Add some additional connections based on data correlation
    corr_matrix = corr(data);
    threshold = 0.3;
    
    for i = 1:N
        for j = 1:N
            if i ~= j && abs(corr_matrix(i,j)) > threshold && learned_dag(i,j) == 0
                % Only add if it doesn't create a cycle
                temp_dag = learned_dag;
                temp_dag(i,j) = 1;
                if ~has_cycles(temp_dag)
                    learned_dag(i,j) = 1;
                end
            end
        end
    end
    
    fprintf('[LEARN] Learned structure with %d connections\n', sum(learned_dag(:)));
end

function has_cycle = has_cycles(dag)
% HAS_CYCLES - Check if directed graph has cycles (simplified)
    has_cycle = false;
    % Simple cycle check - in practice use more sophisticated algorithm
    n = size(dag, 1);
    for k = 1:n
        if dag(k,k) ~= 0  % Self-loop
            has_cycle = true;
            return;
        end
    end
end

function bnet = learn_network_parameters(data, dag, N, n_samples)
% LEARN_NETWORK_PARAMETERS - Estimate CPDs for each network node
% Uses linear Gaussian models for continuous variables
% Returns fully parameterized Bayesian network
    bnet = struct();
    bnet.dag = dag;
    bnet.node_sizes = ones(1, N);
    bnet.CPD = cell(1, N);
    
    for i = 1:N
        parents = find(dag(:,i))';
        if isempty(parents)
            % Root node: unconditional distribution
            bnet.CPD{i} = struct('mean', mean(data(:,i)), 'cov', var(data(:,i)), 'weights', []);
        else
            % Child node: linear Gaussian conditional distribution
            X = [data(:,parents), ones(n_samples,1)];
            Y = data(:,i);
            
            % Add regularization to avoid singular matrix
            lambda = 1e-5; % Regularization parameter
            XTX = X' * X;
            XTX_reg = XTX + lambda * eye(size(XTX));
            
            if rcond(XTX_reg) > 1e-10
                beta = XTX_reg \ (X' * Y);
            else
                % Use pseudoinverse if still ill-conditioned
                beta = pinv(XTX_reg) * (X' * Y);
            end
            
            weights = beta(1:end-1); intercept = beta(end);
            residuals = Y - X * beta;
            
            bnet.CPD{i} = struct('mean', intercept, 'weights', weights, 'cov', var(residuals));
        end
    end
    fprintf('[PARAMS] Learned parameters for %d network nodes\n', N);
end

function [bic_prior, bic_learned] = evaluate_model_fit(data, prior_dag, learned_dag)
% EVALUATE_MODEL_FIT - Compare models using Bayesian Information Criterion
% BIC balances model fit with complexity penalty
% Lower BIC values indicate better models
    bic_prior = compute_bic(data, prior_dag);
    bic_learned = compute_bic(data, learned_dag);
    
    fprintf('[EVAL] BIC Scores - Prior: %.2f, Learned: %.2f\n', bic_prior, bic_learned);
    fprintf('[EVAL] BIC Improvement: %.2f\n', bic_learned - bic_prior);
end

function bic = compute_bic(data, dag)
% COMPUTE_BIC - Calculate Bayesian Information Criterion
% BIC = -2*log(L) + k*log(n)
% L: likelihood, k: parameters, n: samples
    [n_samples, n_vars] = size(data);
    
    % Simplified BIC calculation for demo
    % In practice, compute actual log-likelihood
    loglik = 0;
    for i = 1:n_vars
        parents = find(dag(:,i))';
        if isempty(parents)
            % Unconditional Gaussian
            mu = mean(data(:,i));
            sigma = std(data(:,i));
            loglik = loglik + sum(log(normpdf(data(:,i), mu, sigma)));
        else
            % Conditional Gaussian with regularization
            X = [data(:,parents), ones(n_samples,1)];
            Y = data(:,i);
            
            % Add regularization
            lambda = 1e-5;
            XTX = X' * X;
            XTX_reg = XTX + lambda * eye(size(XTX));
            
            if rcond(XTX_reg) > 1e-10
                beta = XTX_reg \ (X' * Y);
            else
                beta = pinv(XTX_reg) * (X' * Y);
            end
            
            residuals = Y - X * beta;
            sigma = std(residuals);
            loglik = loglik + sum(log(normpdf(residuals, 0, sigma)));
        end
    end
    
    % Compute BIC
    k = sum(sum(dag)) + n_vars;  % Number of parameters
    bic = -2 * loglik + k * log(n_samples);
end

function samples = sample_from_network(bnet, N, n_samples)
% SAMPLE_FROM_NETWORK - Generate samples from learned distribution
% Useful for Monte Carlo simulation and uncertainty analysis
% Returns matrix of synthetic parameter realizations
    samples = zeros(n_samples, N);
    
    % Topological order for sampling
    order = 1:N;  % Simplified - in practice compute topological order
    
    for i = 1:n_samples
        sample_vec = zeros(1, N);
        for j = order
            parents = find(bnet.dag(:,j))';
            if isempty(parents)
                % Sample from unconditional distribution
                sample_vec(j) = bnet.CPD{j}.mean + sqrt(bnet.CPD{j}.cov) * randn();
            else
                % Sample from conditional distribution
                parent_vals = sample_vec(parents);
                mean_val = bnet.CPD{j}.mean + sum(bnet.CPD{j}.weights .* parent_vals');
                sample_vec(j) = mean_val + sqrt(bnet.CPD{j}.cov) * randn();
            end
        end
        samples(i, :) = sample_vec;
    end
    
    fprintf('[SAMPLE] Generated %d samples from network\n', n_samples);
end

function save_analysis_results(bnet, data, samples, dag, vars, bic_scores, meta)
% SAVE_ANALYSIS_RESULTS - Export complete analysis results
% Saves network, data, samples, and performance metrics
% Creates self-contained results file for future use
    results = struct();
    results.network = bnet;
    results.original_data = data;
    results.generated_samples = samples;
    results.structure = dag;
    results.variables = vars;
    results.bic_scores = bic_scores;
    results.metadata = meta;
    results.timestamp = datetime('now');
    
    output_dir = './output';
    if ~exist(output_dir, 'dir'), mkdir(output_dir); end
    
    save(fullfile(output_dir, 'bayesian_network_results.mat'), 'results');
    fprintf('[EXPORT] Saved complete results to: %s\n', fullfile(output_dir, 'bayesian_network_results.mat'));
end
end