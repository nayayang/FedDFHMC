function [ACC, NMI, ARI, Purity, FScore] = FedDFHMC_five_metrics(labels, prediction)
%FEDDFHMC_FIVE_METRICS Score final clustering labels with five scalar metrics.
%   This helper is intended for evaluation only, after clustering has
%   produced PREDICTION.  It does not participate in the clustering step.

if ~localIsFiniteColumn(labels)
    error('FedDFHMC_five_metrics:InvalidLabels', ...
        'labels must be a nonempty finite real numeric column vector.');
end
if ~localIsFiniteColumn(prediction)
    error('FedDFHMC_five_metrics:InvalidPrediction', ...
        'prediction must be a nonempty finite real numeric column vector.');
end
if numel(labels) ~= numel(prediction)
    error('FedDFHMC_five_metrics:InvalidLabels', ...
        'labels and prediction must have equal lengths.');
end

[~, ~, labelIndex] = unique(labels);
[~, ~, predictionIndex] = unique(prediction);
classCount = max(labelIndex);
predictionCount = max(predictionIndex);
sampleCount = numel(labels);
contingency = accumarray([labelIndex, predictionIndex], 1, ...
    [classCount, predictionCount]);

assignment = localMaximumAssignment(contingency);
matchedCount = 0;
for classIndex = 1:classCount
    predictionClass = assignment(classIndex);
    if predictionClass <= predictionCount
        matchedCount = matchedCount + contingency(classIndex, predictionClass);
    end
end
ACC = matchedCount / sampleCount;
Purity = sum(max(contingency, [], 1)) / sampleCount;
NMI = localNMI(contingency, sampleCount);
ARI = localARI(contingency, sampleCount);
FScore = localMacroF1(contingency, assignment, labelIndex, predictionIndex, ...
    classCount, predictionCount);
end

function valid = localIsFiniteColumn(values)
valid = isnumeric(values) && isreal(values) && isvector(values) && ...
    size(values, 2) == 1 && ~isempty(values) && all(isfinite(values));
end

function nmi = localNMI(contingency, sampleCount)
joint = contingency / sampleCount;
labelMarginal = sum(joint, 2);
predictionMarginal = sum(joint, 1);
expected = labelMarginal * predictionMarginal;
nonzero = joint > 0;
mutualInformation = sum(joint(nonzero) .* log2(joint(nonzero) ./ expected(nonzero)));
labelEntropy = -sum(labelMarginal(labelMarginal > 0) .* log2(labelMarginal(labelMarginal > 0)));
predictionEntropy = -sum(predictionMarginal(predictionMarginal > 0) .* ...
    log2(predictionMarginal(predictionMarginal > 0)));
normalizer = max(labelEntropy, predictionEntropy);
if normalizer == 0
    nmi = 1;
else
    nmi = mutualInformation / normalizer;
end
nmi = min(1, max(0, real(nmi)));
end

function ari = localARI(contingency, sampleCount)
if sampleCount < 2
    ari = 1;
    return;
end

pairCount = @(counts) sum(counts .* (counts - 1) / 2);
agreementPairs = pairCount(contingency(:));
labelPairs = pairCount(sum(contingency, 2));
predictionPairs = pairCount(sum(contingency, 1));
totalPairs = sampleCount * (sampleCount - 1) / 2;
expectedAgreement = labelPairs * predictionPairs / totalPairs;
maximumAgreement = (labelPairs + predictionPairs) / 2;
denominator = maximumAgreement - expectedAgreement;

if denominator == 0
    ari = double(agreementPairs == maximumAgreement);
else
    ari = (agreementPairs - expectedAgreement) / denominator;
end
ari = min(1, max(-1, ari));
end

function fscore = localMacroF1(contingency, assignment, labelIndex, predictionIndex, ...
    classCount, predictionCount)
predictionToClass = zeros(predictionCount, 1);
for classIndex = 1:classCount
    predictionClass = assignment(classIndex);
    if predictionClass <= predictionCount
        predictionToClass(predictionClass) = classIndex;
    end
end

mappedPrediction = zeros(numel(predictionIndex), 1);
hasMappedClass = predictionToClass(predictionIndex) > 0;
mappedPrediction(hasMappedClass) = predictionToClass(predictionIndex(hasMappedClass));
f1ByClass = zeros(classCount, 1);
for classIndex = 1:classCount
    truePositive = sum(labelIndex == classIndex & mappedPrediction == classIndex);
    falsePositive = sum(labelIndex ~= classIndex & mappedPrediction == classIndex);
    falseNegative = sum(labelIndex == classIndex & mappedPrediction ~= classIndex);
    denominator = 2 * truePositive + falsePositive + falseNegative;
    f1ByClass(classIndex) = 2 * truePositive / denominator;
end
fscore = mean(f1ByClass);
end

function assignment = localMaximumAssignment(weights)
% Pad to a square matrix, then minimize the complement of the weights.
dimension = max(size(weights));
squareWeights = zeros(dimension);
squareWeights(1:size(weights, 1), 1:size(weights, 2)) = weights;
cost = max(squareWeights(:)) - squareWeights;
assignment = localHungarianMinimum(cost);
end

function assignment = localHungarianMinimum(cost)
dimension = size(cost, 1);
u = zeros(dimension + 1, 1);
v = zeros(dimension + 1, 1);
p = zeros(dimension + 1, 1);
way = zeros(dimension + 1, 1);

for row = 1:dimension
    p(1) = row;
    column0 = 1;
    minimumValues = inf(dimension + 1, 1);
    used = false(dimension + 1, 1);
    while true
        used(column0) = true;
        row0 = p(column0);
        delta = inf;
        column1 = 0;
        for column = 2:dimension + 1
            if ~used(column)
                reducedCost = cost(row0, column - 1) - u(row0 + 1) - v(column);
                if reducedCost < minimumValues(column)
                    minimumValues(column) = reducedCost;
                    way(column) = column0;
                end
                if minimumValues(column) < delta
                    delta = minimumValues(column);
                    column1 = column;
                end
            end
        end
        for column = 1:dimension + 1
            if used(column)
                u(p(column) + 1) = u(p(column) + 1) + delta;
                v(column) = v(column) - delta;
            else
                minimumValues(column) = minimumValues(column) - delta;
            end
        end
        column0 = column1;
        if p(column0) == 0
            break;
        end
    end
    while true
        column1 = way(column0);
        p(column0) = p(column1);
        column0 = column1;
        if column0 == 1
            break;
        end
    end
end

assignment = zeros(dimension, 1);
for column = 2:dimension + 1
    assignment(p(column)) = column - 1;
end
end
