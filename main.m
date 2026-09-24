clc;
clear;

%% Paths
root = fileparts(mfilename('fullpath'));

algorithmDir = fullfile(root, 'algorithm');
secaggDir = fullfile(root, 'secure_aggregation_reference');
gspboxDir = fullfile(root, 'gspbox');
dataPath = fullfile(root, 'data', 'HW2sources.mat');

if ~isfolder(algorithmDir) || ~isfolder(secaggDir) || ~isfolder(gspboxDir)
    error('Required dependency directory is missing.');
end
if ~isfile(dataPath)
    error('HW2sources.mat was not found.');
end

addpath(algorithmDir);
addpath(secaggDir);
addpath(genpath(gspboxDir));

%% Load data
S = load(dataPath);
if ~isfield(S, 'data') || ~iscell(S.data) || isempty(S.data)
    error('run_FedDFHMC_HW2sources:InvalidData', ...
        'HW2sources.mat must contain a nonempty cell array named data.');
end
XCell = S.data(:);
numClusters = 10;
n = [];
for v = 1:numel(XCell)
    Xv = XCell{v};
    if ~isnumeric(Xv) || ~isreal(Xv) || ~ismatrix(Xv) || isempty(Xv) || ...
            any(~isfinite(Xv(:)))
        error('run_FedDFHMC_HW2sources:InvalidView', ...
            'data{%d} must be a nonempty, real, finite numeric matrix.', v);
    end
    if v == 1
        n = max(size(Xv));
    end
    if size(Xv, 2) ~= n
        if size(Xv, 1) == n
            Xv = Xv.';
        else
            error('run_FedDFHMC_HW2sources:InconsistentSampleCount', ...
                'data{%d} cannot be oriented with %d samples.', v, n);
        end
    end
    columnNorms = sqrt(sum(Xv.^2, 1));
    if any(~isfinite(columnNorms)) || any(columnNorms <= 0)
        error('run_FedDFHMC_HW2sources:InvalidViewNormalization', ...
            'data{%d} contains a zero-norm or non-finite sample.', v);
    end
    XCell{v} = Xv ./ columnNorms;
end

if ~isfield(S, 'truelabel') || ~iscell(S.truelabel) || isempty(S.truelabel)
    error('run_FedDFHMC_HW2sources:InvalidLabels', ...
        'HW2sources.mat must contain truelabel{1}.');
end
[~, ~, labels] = unique(S.truelabel{1}(:));
if numel(labels) ~= n
    error('run_FedDFHMC_HW2sources:InconsistentLabelCount', ...
        'The label count does not match the sample count.');
end

%% Parameters
param = struct();
param.hg_k = 5;
param.maxIter = 20;
param.rho = 20;
param.gamma = 2;
param.beta = 1.2;
param.tau = 0.03;
param.lambda_z = 0.001;
param.m = 2;
param.hiddenDims = 50;
param.k = numClusters;
param.innerH = 10;
param.innerZ = 1;
param.local_ridge_lambda = 0.1;
param.local_ridge_kmeans_replicates = 10;
param.local_ridge_kmeans_maxit = 100;
param.local_ridge_solver_maxit = 200;
param.local_ridge_solver_tol = 1e-8;
param.fixed_iterations = false;
param.tolObj = 1e-3;
param.tolPri = 5e-3;
param.tolDual = 5e-3;
param.stop_patience = 2;
param.use_aead_secure_aggregation = true;
param.secagg_modulus = int64(2147483647);
param.secagg_scale = 1e6;
param.secagg_clip_threshold = 500;
param.verbose = 0;

%% Hypergraph construction
% Construct the label-free hypergraphs once and reuse them in all runs.
probeParam = param;
probeParam.seed = 1;
probeParam.maxIter = 1;
[~, ~, ~, ~, ~, LCell, usedGspHypergraph] = ...
    FedDFHMC(XCell, [], probeParam);
assert(all(usedGspHypergraph), ...
    'run_FedDFHMC_HW2sources:HypergraphUnavailable', ...
    'The required GSPBox hypergraph Laplacian was not used.');
LCell = LCell(:);

%% Training and evaluation
trainingSeeds = 1;
metricNames = {'ACC', 'NMI', 'ARI', 'Purity', 'FScore'};
metricValues = zeros(numel(trainingSeeds), numel(metricNames));
elapsedSeconds = zeros(numel(trainingSeeds), 1);
finalObjective = zeros(numel(trainingSeeds), 1);
bestKMeansInertia = zeros(numel(trainingSeeds), 1);

trainingSeed = trainingSeeds(1);
param.seed = trainingSeed;
fprintf('Training seed %d ...\n', trainingSeed);
trainTimer = tic;
[H, ~, ~, ~, Obj] = FedDFHMC(XCell, LCell, param);
elapsedSeconds(1) = toc(trainTimer);
if isempty(Obj) || any(~isfinite(Obj(:))) || ...
        ~isequal(size(H), [numClusters, n]) || any(~isfinite(H(:)))
    error('run_FedDFHMC_HW2sources:InvalidTrainingOutput', ...
        'Training seed %d returned invalid output.', trainingSeed);
end

embedding = H.';
rowNorms = sqrt(sum(embedding.^2, 2));
if any(~isfinite(rowNorms)) || any(rowNorms <= 0)
    error('run_FedDFHMC_HW2sources:InvalidEmbedding', ...
        'Training seed %d produced a zero-norm sample embedding.', trainingSeed);
end
embedding = embedding ./ rowNorms;
[bestPrediction, bestKMeansInertia(1)] = select_kmeans_partition( ...
    embedding, numClusters, trainingSeed);
[metricValues(1,1), metricValues(1,2), metricValues(1,3), ...
    metricValues(1,4), metricValues(1,5)] = ...
    FedDFHMC_five_metrics(labels, bestPrediction);
finalObjective(1) = Obj(end);

resultTable = array2table( ...
    [trainingSeeds, metricValues, elapsedSeconds, finalObjective, ...
     bestKMeansInertia], ...
    'VariableNames', [{'TrainingSeed'}, metricNames, ...
    {'ElapsedSeconds', 'FinalObjective', 'BestKMeansInertia'}]);

%% Save results
resultsDir = fullfile(root, 'results');
if ~isfolder(resultsDir)
    [created, message] = mkdir(resultsDir);
    if ~created
        error('run_FedDFHMC_HW2sources:ResultsDirectoryCreationFailed', ...
            'Could not create %s: %s', resultsDir, message);
    end
end

resultCsv = fullfile(resultsDir, 'HW2sources_seed1_result.csv');
resultMat = fullfile(resultsDir, 'HW2sources_results.mat');
writetable(resultTable, resultCsv);

save(resultMat, 'param', 'LCell', 'trainingSeeds', 'metricNames', ...
    'metricValues', 'elapsedSeconds', 'finalObjective', ...
    'bestKMeansInertia', 'resultTable', '-v7.3');

for metricIndex = 1:numel(metricNames)
    fprintf('%s: %.4f\n', metricNames{metricIndex}, metricValues(metricIndex));
end

function [bestPrediction, bestInertia] = select_kmeans_partition( ...
        embedding, numClusters, trainingSeed)
% Select the best of ten deterministic K-means++ runs by inertia.
bestPrediction = [];
bestInertia = inf;
for replicate = 1:10
    rng(100000*trainingSeed+replicate, 'twister');
    [prediction, centers] = FedDFHMC_lloyd_kmeans( ...
        embedding, numClusters, 1000);
    prediction = prediction(:);
    residuals = embedding-centers(prediction,:);
    inertia = sum(residuals.^2, 'all');
    if isfinite(inertia) && inertia < bestInertia
        bestInertia = inertia;
        bestPrediction = prediction;
    end
end
if isempty(bestPrediction)
    error('run_FedDFHMC_HW2sources:NoKMeansResult', ...
        'No valid K-means result was produced for training seed %d.', ...
        trainingSeed);
end
end
