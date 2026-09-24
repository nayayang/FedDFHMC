% #### Clustering evaluation ####
function [idx, centers] = FedDFHMC_lloyd_kmeans(X, K, maxIter)
% Serial Lloyd k-means with k-means++ initialization.
%
% Inputs and outputs:
%   X       : n-by-d feature matrix, one sample per row.
%   K       : number of clusters.
%   maxIter : maximum Lloyd iterations.
%   idx     : n-by-1 cluster assignment vector in 1..K.
%   centers : K-by-d cluster center matrix.

n = size(X,1);
if K < 1 || K > n, error('FedDFHMC_lloyd_kmeans:InvalidK', 'Invalid cluster count.'); end

centers = initialize_kmeans_pp(X, K);
idx = zeros(n,1);

for it = 1:maxIter
    d2 = sum(X.^2,2) + sum(centers.^2,2)' - 2*(X*centers');
    d2 = max(d2, 0);

    [~, nextIdx] = min(d2, [], 2);

    if it > 1 && isequal(nextIdx, idx), break; end
    idx = nextIdx;


    for c = 1:K
        members = X(idx == c,:);
        if isempty(members)
            [~, farthest] = max(min(d2, [], 2));
            centers(c,:) = X(farthest,:);
        else
            centers(c,:) = mean(members, 1);
        end
    end
end
end

function centers = initialize_kmeans_pp(X, K)
n = size(X,1);
centers = zeros(K, size(X,2));
chosen = false(n,1);


first = randi(n);
centers(1,:) = X(first,:);
chosen(first) = true;
closestD2 = sum((X - centers(1,:)).^2, 2);

for c = 2:K
    total = sum(closestD2);
    if ~isfinite(total) || total <= eps
        candidates = find(~chosen);
        pick = candidates(randi(numel(candidates)));
    else
        threshold = rand()*total;
        pick = find(cumsum(closestD2) >= threshold, 1, 'first');
        if isempty(pick) || chosen(pick)
            candidates = find(~chosen);
            [~, localPick] = max(closestD2(candidates));
            pick = candidates(localPick);
        end
    end

    centers(c,:) = X(pick,:);
    chosen(pick) = true;


    newD2 = sum((X - centers(c,:)).^2, 2);
    closestD2 = min(closestD2, newD2);
end
end
