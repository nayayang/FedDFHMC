function [H, Hlocal, Z, alpha, Obj, LCell, usedGspHypergraph] = FedDFHMC(XCell, LCell, param)
% FedDFHMC
% Federated Deep Factorization + Hypergraph Regularization + Adaptive Multi-view Consensus
%
% This implements an ADMM-style consensus on the latent representation H:
%
%   min_{ {Z_l^(v)}, {H^(v)}, H, alpha }  sum_v (alpha(v))^gamma * ( ||X^(v) - W^(v) H^(v)||_F^2
%                                          + beta * tr( H^(v) L^(v) H^(v)' ) )
%                                      + tau * ||H||_* + lambda_z * sum_{v,l} ||Z_l^(v)||_F^2
%   s.t.  H^(v) >= 0,  H^(v) = H,  sum_v alpha(v)=1, alpha(v)>=0
%
% Notes:
%   - Each view v is treated as one client. XCell{v} stays on the client; only H/dual are aggregated.
%   - LCell{v} should be an n-by-n hypergraph Laplacian. If LCell is empty,
%     it is constructed locally for each view using GSPBox gsp_nn_hypergraph.
%   - H^(v) is updated with monotone accelerated projected gradient.
%   - Global H is updated by the exact Singular Value Thresholding step in Eq. (17).
%
% Inputs:
%   XCell  : Vx1 cell, XCell{v} is d_v-by-n
%   LCell  : Vx1 cell, LCell{v} is n-by-n (can be sparse). Pass [] to build from X.
%   param  : struct with fields (recommended defaults below)
%
% Outputs:
%   H      : k-by-n global consensus representation
%   Hlocal : Vx1 cell, local H^(v)
%   Z      : Vx1 cell, each is a 1-by-m cell of layer matrices Z_l^(v)
%   alpha  : V-by-1 adaptive weights
%   Obj    : objective value per communication round
%   LCell  : Vx1 cell containing the local hypergraph Laplacians
%   usedGspHypergraph : Vx1 logical vector reporting GSPBox construction
%
% Recommended param fields:
%   param.maxIter     default 20
%   param.rho         ADMM penalty for H^(v)=H 
%   param.gamma       exponent on alpha, >1 
%   param.beta        hypergraph regularization weight 
%   param.tau         nuclear norm weight 
%   param.lambda_z    Frobenius reg on the basis layers Z_l^(v) 
%   param.m           number of Z layers 
%   param.hiddenDims  1-by-(m-1) hidden dimensions (default 50 for each hidden layer)
%   param.k           latent dimension k 
%   param.innerH      projected GD steps for H^(v) per outer iter 
%   param.innerZ      alternating closed-form sweeps per Z layer per outer iter 
%   param.local_ridge_lambda ridge coefficient for local soft-code pretraining 
%   param.local_ridge_kmeans_replicates local dictionary starts 
%   param.local_ridge_kmeans_maxit local Lloyd iterations 
%   param.local_ridge_solver_maxit projected-gradient iterations 
%   param.local_ridge_solver_tol projected-gradient tolerance 
%   param.fixed_iterations disable convergence-based early stopping 
%   param.hg_sigma_scale multiplier on the automatic per-view hypergraph sigma 
%   param.verbose    
%

%% defaults
if nargin < 3, param = struct(); end
V = length(XCell);
n = size(XCell{1}, 2);

if ~isfield(param,'maxIter'),   param.maxIter = 20; end
if ~isfield(param,'rho'),       param.rho = 10; end
if ~isfield(param,'gamma'),     param.gamma = 2; end
if ~isfield(param,'beta'),      param.beta = 1; end
if ~isfield(param,'tau'),       param.tau = 1; end
if ~isfield(param,'lambda_z'),  param.lambda_z = 1e-3; end
if ~isfield(param,'m'),         param.m = 2; end
if ~isscalar(param.m) || ~isfinite(param.m) || param.m < 2 || param.m ~= floor(param.m)
    error('param.m must be an integer greater than or equal to 2.');
end
if ~isfield(param,'hiddenDims'),param.hiddenDims = 50*ones(1,param.m-1); end
if ~isfield(param,'innerH'),    param.innerH = 10; end
if ~isfield(param,'innerZ'),    param.innerZ = 1; end
if ~isfield(param,'hg_k'),     param.hg_k = 5; end
if ~isfield(param,'hg_sigma'), param.hg_sigma = []; end
if ~isfield(param,'hg_sigma_scale'), param.hg_sigma_scale = 1; end
if ~isfield(param,'verbose'),   param.verbose = 1; end
if ~isfield(param,'seed'),      param.seed = 1; end
if ~isfield(param,'local_ridge_lambda'), param.local_ridge_lambda = 0.1; end
if ~isfield(param,'local_ridge_kmeans_replicates'), param.local_ridge_kmeans_replicates = 10; end
if ~isfield(param,'local_ridge_kmeans_maxit'), param.local_ridge_kmeans_maxit = 100; end
if ~isfield(param,'local_ridge_solver_maxit'), param.local_ridge_solver_maxit = 200; end
if ~isfield(param,'local_ridge_solver_tol'), param.local_ridge_solver_tol = 1e-8; end

if ~isfield(param,'max_backtracking'), param.max_backtracking = 20; end
if ~isfield(param,'backtracking_factor'), param.backtracking_factor = 0.5; end
if ~isfield(param,'armijo_c'), param.armijo_c = 1e-4; end

if ~isfield(param,'fixed_iterations'), param.fixed_iterations = false; end
% Unsupervised adaptive stopping is evaluated from outer iteration 2 onward.
if ~isfield(param,'tolObj'), param.tolObj = 1e-3; end
if ~isfield(param,'tolPri'), param.tolPri = 5e-3; end
if ~isfield(param,'tolDual'), param.tolDual = 5e-3; end
if ~isfield(param,'stop_patience'), param.stop_patience = 2; end


% Secure aggregation defaults. These parameters affect only the server-side
% average Qbar=(1/V)*sum_v(Hlocal{v}+Y{v}/rho) used before the SVT step.

if ~isfield(param,'use_aead_secure_aggregation'), param.use_aead_secure_aggregation = false; end
if ~isfield(param,'secagg_modulus'), param.secagg_modulus = int64(2147483647); end
if ~isfield(param,'secagg_scale'), param.secagg_scale = 1e6; end
if ~isfield(param,'secagg_clip_threshold'), param.secagg_clip_threshold = []; end
if ~isscalar(param.hg_sigma_scale) || ~isfinite(param.hg_sigma_scale) || param.hg_sigma_scale <= 0
    error('param.hg_sigma_scale must be a positive finite scalar.');
end
if ~isscalar(param.local_ridge_lambda) || ~isfinite(param.local_ridge_lambda) || ...
        param.local_ridge_lambda <= 0
    error('param.local_ridge_lambda must be a positive finite scalar.');
end
if param.local_ridge_kmeans_replicates < 1 || ...
        param.local_ridge_kmeans_replicates ~= floor(param.local_ridge_kmeans_replicates)
    error('param.local_ridge_kmeans_replicates must be a positive integer.');
end
if param.local_ridge_kmeans_maxit < 1 || ...
        param.local_ridge_kmeans_maxit ~= floor(param.local_ridge_kmeans_maxit)
    error('param.local_ridge_kmeans_maxit must be a positive integer.');
end
if param.local_ridge_solver_maxit < 1 || ...
        param.local_ridge_solver_maxit ~= floor(param.local_ridge_solver_maxit)
    error('param.local_ridge_solver_maxit must be a positive integer.');
end
if ~isscalar(param.local_ridge_solver_tol) || ...
        ~isfinite(param.local_ridge_solver_tol) || param.local_ridge_solver_tol <= 0
    error('param.local_ridge_solver_tol must be a positive finite scalar.');
end

rng(param.seed);

% latent dim k
if isfield(param,'k')
    k = param.k;
else
    k = 20;
end

m = param.m;
hidden = param.hiddenDims(:)';
if numel(hidden) ~= m-1 || any(~isfinite(hidden)) || ...
        any(hidden < 1) || any(hidden ~= floor(hidden))
    error('param.hiddenDims must contain m-1 positive integers.');
end

%% build / validate Laplacians

if nargin < 2 || isempty(LCell)
    % Build one GSPBox hypergraph Laplacian per view.
    LCell = cell(V,1);
    usedGspHypergraph = false(V,1);
    hypergraphSigma = nan(V,1);
    for v = 1:V
        [LCell{v}, usedGspHypergraph(v), hypergraphSigma(v)] = ...
            build_hypergraph_laplacian(XCell{v}, param);
    end
else
    if length(LCell) ~= V
        error('LCell must be a Vx1 cell with same number of views as XCell.');
    end
    usedGspHypergraph = false(V,1);
    hypergraphSigma = nan(V,1);
end

reconstructionScale = ones(V,1);
laplacianScale = ones(V,1);

%% init variables

alpha = ones(V,1) / V;

H = zeros(k,n);

% layer dims per view: d_v -> hidden... -> k
Z = cell(V,1);
W = cell(V,1);
Hlocal = cell(V,1);
Y = cell(V,1); % duals for H^(v)=H

localSoftCodes = cell(V,1);
for v = 1:V
    localSoftCodes{v} = ridge_local_soft_code(XCell{v}, k, param);
end
sharedInitH = mean(cat(3, localSoftCodes{:}), 3);

for v = 1:V
    d = size(XCell{v}, 1);
    dims = [d, hidden, k]; % length m+1
    [Zv, Hv0] = initialize_deep_factors(dims, sharedInitH);
    Z{v} = Zv;
    W{v} = prodZ(Zv);

    Hlocal{v} = Hv0;
    Y{v} = zeros(k,n);
end

for v = 1:V
    H = H + Hlocal{v};
end
H = H / V;

Obj = zeros(param.maxIter,1);
primalRes = zeros(param.maxIter,1);
dualRes = zeros(param.maxIter,1);
relativeObjChange = nan(param.maxIter,1);
normalizedPrimalRes = zeros(param.maxIter,1);
normalizedDualRes = zeros(param.maxIter,1);
rhoHistory = zeros(param.maxIter,1);
stopSatisfiedStreak = 0;

secureAggContext = struct();
if param.use_aead_secure_aggregation
    % The AEAD aggregation protocol is implemented in role-specific modules.
    if exist('secure_aggregation_reference', 'file') ~= 2 || ...
            exist('secagg_key_manager', 'file') ~= 2 || ...
            exist('secagg_server_create_downlink', 'file') ~= 2 || ...
            exist('secagg_client_receive_downlink', 'file') ~= 2
        error(['AEAD secure aggregation is enabled, but ', ...
            'secure_aggregation_reference/ is not on the MATLAB path.']);
    end
    secureAggContext = secagg_key_manager(V);
end

%% outer loop
for it = 1:param.maxIter
    [Z, W, Hlocal] = ...
        update_all_clients(XCell, LCell, Z, Hlocal, alpha, ...
        reconstructionScale, laplacianScale, param, H, Y);

    % ---- global/server update for H via SVT ----
    % H = argmin tau||H||_* + (rho/2) * sum_v ||Hlocal{v} - H + Y{v}/rho||_F^2
    % => SVT on average

    Hbar = aggregate_global_code( ...
        Hlocal, Y, param, it, secureAggContext, H);
    Hprev = H;

    H = svt(Hbar, param.tau/(param.rho*V));

    % ---- dual update ----
    Y = update_dual_variables(Y, Hlocal, H, param.rho);

    % ---- alpha update ----
    E = compute_view_errors(XCell, LCell, W, Hlocal, ...
        reconstructionScale, laplacianScale, param);
    % alpha(v) ∝ E(v)^(1/(1-gamma))
    pow = 1/(1 - param.gamma);
    tmp = E.^pow;
    alphaClosed = tmp / sum(tmp);
    alpha = alphaClosed;

    send_secure_downlink(secureAggContext, H, alpha, it, ...
        param.use_aead_secure_aggregation);

    % ---- objective and stopping residuals ----
    obj = compute_objective(XCell, LCell, W, Hlocal, Z, alpha, H, ...
        reconstructionScale, laplacianScale, param);

    Obj(it) = obj;

    % residuals
    [primalRes(it), localNormSquared] = consensus_residual(Hlocal, H);
    primalScale = max([1, sqrt(localNormSquared), sqrt(V)*norm(H, 'fro')]);
    normalizedPrimalRes(it) = primalRes(it) / primalScale;

    dualRes(it) = param.rho*norm(H - Hprev,'fro');
    dualScale = max(1, param.rho*sqrt(V)*norm(H, 'fro'));
    normalizedDualRes(it) = dualRes(it) / dualScale;
    if it > 1
        relativeObjChange(it) = abs(Obj(it)-Obj(it-1)) / max(1, abs(Obj(it-1)));
    end
    rhoHistory(it) = param.rho;

    if param.verbose && (mod(it,10)==0 || it==1)

        fprintf('[FedDFHMC] iter %d, Obj=%.6e, primRes=%.3e, dualRes=%.3e, rho=%.3g\n', ...
            it, Obj(it), primalRes(it), dualRes(it), rhoHistory(it));
    end


    if ~param.fixed_iterations && it > 1
        stopThisRound = relativeObjChange(it) < param.tolObj && ...
            normalizedPrimalRes(it) < param.tolPri && ...
            normalizedDualRes(it) < param.tolDual;
        if stopThisRound
            stopSatisfiedStreak = stopSatisfiedStreak + 1;
        else
            stopSatisfiedStreak = 0;
        end
        if stopSatisfiedStreak >= param.stop_patience
            Obj = Obj(1:it);
            primalRes = primalRes(1:it);
            dualRes = dualRes(1:it);
            relativeObjChange = relativeObjChange(1:it);
            normalizedPrimalRes = normalizedPrimalRes(1:it);
            normalizedDualRes = normalizedDualRes(1:it);
            rhoHistory = rhoHistory(1:it);
            break;
        end
    end
end

end

%% ----- helpers (local functions) -----

function Hbar = aggregate_global_code( ...
        Hlocal, Y, param, roundIndex, secureContext, H)
% Aggregate Q^(v)=H^(v)+Y^(v)/rho
V = numel(Hlocal);
if param.use_aead_secure_aggregation
    dualScaled = scale_dual_variables(Y, param.rho);
    Hbar = secure_aggregation_reference( ...
        Hlocal, dualScaled, param.secagg_modulus, param.secagg_scale, ...
        param.secagg_clip_threshold, roundIndex, secureContext);
else
    Hbar = plain_consensus_average(Hlocal, Y, param.rho, H);
end
end

function dualScaled = scale_dual_variables(Y, rho)
dualScaled = cell(size(Y));
for v = 1:numel(Y)
    dualScaled{v} = Y{v}/rho;
end
end

function Hbar = plain_consensus_average(Hlocal, Y, rho, H)
Hbar = zeros(size(H));
for v = 1:numel(Hlocal)
    Hbar = Hbar + Hlocal{v} + Y{v}/rho;
end
Hbar = Hbar/numel(Hlocal);
end

function Y = update_dual_variables(Y, Hlocal, H, rho)
for v = 1:numel(Y)
    Y{v} = Y{v} + rho*(Hlocal{v}-H);
end
end

function errors = compute_view_errors(XCell, LCell, W, Hlocal, ...
        reconstructionScale, laplacianScale, param)
V = numel(XCell);
errors = zeros(V,1);
for v = 1:V
    residual = XCell{v}-W{v}*Hlocal{v};
    reconstruction = sum(residual.^2, 'all');
    graph = trace(Hlocal{v}*LCell{v}*Hlocal{v}');
    errors(v) = reconstruction/reconstructionScale(v) + ...
        param.beta*graph/laplacianScale(v)+1e-12;
end
end

function send_secure_downlink(context, H, alpha, roundIndex, enabled)
if ~enabled
    return;
end
packets = secagg_server_create_downlink(context.server, H, alpha, roundIndex);
for v = 1:numel(context.clients)
    secagg_client_receive_downlink(context.clients{v}, packets{v});
end
end

function value = compute_objective(XCell, LCell, W, Hlocal, Z, alpha, H, ...
        reconstructionScale, laplacianScale, param)
value = 0;
zPenalty = 0;
for v = 1:numel(XCell)
    weight = max(alpha(v),eps)^param.gamma;
    residual = XCell{v}-W{v}*Hlocal{v};
    reconstruction = sum(residual.^2, 'all');
    graph = trace(Hlocal{v}*LCell{v}*Hlocal{v}');
    value = value + weight*(reconstruction/reconstructionScale(v) + ...
        param.beta*graph/laplacianScale(v));
    zPenalty = zPenalty + basis_penalty(Z{v});
end
value = value + param.tau*sum(svd(H,'econ')) + param.lambda_z*zPenalty;
end

function value = basis_penalty(Zlayers)
value = 0;
for layer = 1:numel(Zlayers)
    value = value + sum(Zlayers{layer}.^2, 'all');
end
end

function [primalResidual, localNormSquared] = consensus_residual(Hlocal, H)
residualSquared = 0;
localNormSquared = 0;
for v = 1:numel(Hlocal)
    residualSquared = residualSquared + norm(Hlocal{v}-H, 'fro')^2;
    localNormSquared = localNormSquared + norm(Hlocal{v}, 'fro')^2;
end
primalResidual = sqrt(residualSquared);
end

function [Z, W, Hlocal] = ...
        update_all_clients(XCell, LCell, Z, Hlocal, alpha, ...
        reconstructionScale, laplacianScale, param, H, Y)
V = numel(XCell);
W = cell(V,1);
for v = 1:V
    [Z{v}, W{v}, Hlocal{v}] = ...
        update_one_client(XCell{v}, LCell{v}, Z{v}, Hlocal{v}, ...
        alpha(v), reconstructionScale(v), laplacianScale(v), ...
        param, H, Y{v});
end
end

function [Zv, Wv, Hv] = ...
        update_one_client(Xv, Lv, Zv, Hv, alphaV, reconScale, ...
        laplacianScale, param, H, Yv)
% Perform the complete local update for one client.
weight = max(alphaV, eps)^param.gamma;
reconstructionWeight = weight/reconScale;
graphWeight = param.beta/laplacianScale;
for sweep = 1:param.innerZ
    Zv = update_basis_forward(Zv, Hv, Xv, reconstructionWeight, param.lambda_z);
    Zv = update_basis_backward(Zv, Hv, Xv, reconstructionWeight, param.lambda_z);
    Zv = balance_two_layer_scales(Zv);
end
Wv = prodZ(Zv);
nW = norm(Wv, 2)^2;
nL = norm(full(Lv), 2);
Hv = update_local_h(Hv, Xv, Wv, Lv, ...
    weight, reconScale, graphWeight, param, H, Yv, nW, nL);
end

function Zlayers = update_basis_forward(Zlayers, H, X, weight, lambdaZ)
% Forward Sylvester sweep over the deep basis layers.
for layer = 1:numel(Zlayers)
    [A, B] = prepost_products(Zlayers, layer, H);
    Zlayers{layer} = solve_basis_sylvester(A, B, X, weight, lambdaZ);
end
end

function Zlayers = update_basis_backward(Zlayers, H, X, weight, lambdaZ)
% Backward Sylvester sweep over the deep basis layers.
for layer = numel(Zlayers):-1:1
    [A, B] = prepost_products(Zlayers, layer, H);
    Zlayers{layer} = solve_basis_sylvester(A, B, X, weight, lambdaZ);
end
end

function W = prodZ(Zlayers)
% Return the product of all deep basis layers for one view.
% If Zlayers = {Z1,Z2,...,Zm}, then W = Z1*Z2*...*Zm.
W = Zlayers{1};
for i = 2:length(Zlayers)
    W = W*Zlayers{i};
end
end


function [Zlayers, H0] = initialize_deep_factors(dims, sharedInitH)
% Initialize every deep basis layer randomly and use the shared local code.
numLayers = numel(dims)-1;
if size(sharedInitH,1) ~= dims(end)
    error('Local ridge initialization has an invalid latent dimension.');
end
Zlayers = cell(1,numLayers);
for layer = 1:numLayers
    Zlayers{layer} = 0.1*randn(dims(layer),dims(layer+1));
end
H0 = sharedInitH;
end



function Hsoft = ridge_local_soft_code(X, k, param)
% Learn a local nonnegative soft code without labels or centralized data.
%
%   min_z 0.5*||x_i-A*z||_2^2 + 0.5*lambda*||z||_2^2
%   s.t. z >= 0, 1'*z = 1,
%
% using projected gradient on the probability simplex. 

if param.local_ridge_lambda <= 0
    error('param.local_ridge_lambda must be positive.');
end
features = X.';
bestInertia = inf;
bestCenters = [];
for replicate = 1:param.local_ridge_kmeans_replicates
    [assignment, centers] = FedDFHMC_lloyd_kmeans( ...
        features, k, param.local_ridge_kmeans_maxit);
    residuals = features - centers(assignment,:);
    inertia = sum(residuals.^2, 'all');
    if isfinite(inertia) && inertia < bestInertia
        bestInertia = inertia;
        bestCenters = centers;
    end
end
if isempty(bestCenters)
    error('FedDFHMC:LocalRidgePretrainingFailed', ...
        'Local ridge pretraining could not produce a finite dictionary.');
end

A = bestCenters.';
hessian = A.'*A + param.local_ridge_lambda*eye(k);
linearTerm = A.'*X;
step = 1/max(norm(hessian, 2), eps);
Hsoft = ones(k, size(X,2))/k;
for iteration = 1:param.local_ridge_solver_maxit
    nextH = project_columns_to_simplex( ...
        Hsoft - step*(hessian*Hsoft - linearTerm));
    relativeChange = norm(nextH-Hsoft, 'fro')/max(1, norm(Hsoft, 'fro'));
    Hsoft = nextH;
    if relativeChange < param.local_ridge_solver_tol
        break;
    end
end
end


function projected = project_columns_to_simplex(values)
% Euclidean projection of every column onto {z >= 0, sum(z)=1}.
[rows, columns] = size(values);
projected = zeros(rows, columns);
for column = 1:columns
    sortedValues = sort(values(:,column), 'descend');
    cumulative = cumsum(sortedValues) - 1;
    active = find(sortedValues - cumulative./(1:rows)' > 0, 1, 'last');
    threshold = cumulative(active)/active;
    projected(:,column) = max(values(:,column)-threshold, 0);
end
end



function Hv = update_local_h(Hv, X, W, L, av, reconScale, betaScaled, param, H, Y, nW, nL)
% Monotone FISTA for the nonnegative local-code subproblem.
% Hv is pulled toward global H by the ADMM penalty and projected to Hv >= 0.
avRecon = av/reconScale;
graphWeight = av*betaScaled;
x = Hv;
y = x;
t = 1;
fX = local_h_objective(x, X, W, L, avRecon, graphWeight, param.rho, H, Y);
Lh = 2*avRecon*nW + 2*graphWeight*nL + param.rho + 1e-12;
initialEta = 1/Lh;
for inner = 1:param.innerH
    grad = local_h_gradient(y, X, W, L, avRecon, graphWeight, param.rho, H, Y);
    [candidate, fCandidate] = projected_h_step(y, grad, initialEta, ...
        X, W, L, avRecon, graphWeight, param, H, Y);

    tolerance = 1e-12*max(1, abs(fX));
    if fCandidate > fX + tolerance
        % The extrapolation was unhelpful. Restart from the last feasible point.
        y = x;
        t = 1;
        grad = local_h_gradient(y, X, W, L, avRecon, graphWeight, param.rho, H, Y);
        [candidate, fCandidate] = projected_h_step(y, grad, initialEta, ...
            X, W, L, avRecon, graphWeight, param, H, Y);
    end

    if fCandidate > fX + tolerance
        candidate = x;
        fCandidate = fX;
    end

    xPrevious = x;
    x = candidate;
    fX = fCandidate;

    tNext = (1 + sqrt(1 + 4*t^2))/2;
    y = x + ((t-1)/tNext)*(x-xPrevious);
    t = tNext;
end
Hv = x;
end


function grad = local_h_gradient(Hv, X, W, L, avRecon, graphWeight, rho, H, Y)
% Gradient of local H objective:
% reconstruction + hypergraph smoothness + ADMM consensus penalty.
grad = 2*(avRecon*W'*(W*Hv-X) + graphWeight*(Hv*L)) + rho*(Hv-H) + Y;
end


function [candidate, fCandidate] = projected_h_step(base, grad, eta, X, W, L, avRecon, graphWeight, param, H, Y)
% Try a projected gradient step and optionally shrink eta by backtracking.
% Projection is max(candidate,0), enforcing H^(v) >= 0.
fBase = local_h_objective(base, X, W, L, avRecon, graphWeight, param.rho, H, Y);
for attempt = 0:param.max_backtracking
    candidate = max(base-eta*grad, 0);
    fCandidate = local_h_objective(candidate, X, W, L, avRecon, graphWeight, param.rho, H, Y);
    step = candidate-base;
    if fCandidate <= fBase + param.armijo_c*sum(grad.*step, 'all')
        return;
    end
    eta = eta*param.backtracking_factor;
end
end


function Zlayers = balance_two_layer_scales(Zlayers)
% Minimize ||c*Z1||_F^2 + ||Z2/c||_F^2 while preserving Z1*Z2.
if numel(Zlayers) ~= 2
    return;
end

norm1 = norm(Zlayers{1}, 'fro');
norm2 = norm(Zlayers{2}, 'fro');
if norm1 <= eps || norm2 <= eps
    return;
end

scale = sqrt(norm2/norm1);
Zlayers{1} = scale*Zlayers{1};
Zlayers{2} = Zlayers{2}/scale;
end


function value = local_h_objective(Hv, X, W, L, avRecon, graphWeight, rho, H, Y)
% Local augmented objective used only for line search and monotone checks.
R = X - W*Hv;
consensus = Hv - H + Y/rho;
value = avRecon*sum(R.^2, 'all') + graphWeight*trace(Hv*L*Hv') ...
    + (rho/2)*sum(consensus.^2, 'all');
end

function [A, B] = prepost_products(Zlayers, l, H)
% A = Z1*...*Z_{l-1}; empty means the identity for the first layer.
% B = Z_{l+1}*...*Z_m*H (or H)
m = length(Zlayers);
if l == 1
    A = [];
else
    A = Zlayers{1};
    for i = 2:(l-1)
        A = A*Zlayers{i};
    end
end
if l == m
    B = H;
else
    T = Zlayers{l+1};
    for i = (l+2):m
        T = T*Zlayers{i};
    end
    B = T*H;
end
end

function Zl = solve_basis_sylvester(A, B, X, av, lambda_z)
% Solve av*A'*A*Zl*B*B' + lambda_z*Zl = av*A'*X*B'.
% This is the elementwise eigenspace update derived in Eq. (5)-(11).
Right = (B * B' + (B * B')') / 2;
[Q, Dright] = eig(full(Right), 'vector');
Dright = max(real(Dright), 0);

if isempty(A)
    % A is the identity for Z1. Avoid a large, redundant eig(I) each sweep.
    C = av*(X*B')*Q;
    denom = av*repmat(Dright(:)', size(X,1), 1) + lambda_z;
    Zl = real((C./max(denom, eps))*Q');
    return;
end

Left = (A' * A + (A' * A)') / 2;
[P, Dleft] = eig(full(Left), 'vector');
Dleft = max(real(Dleft), 0);

C = P' * (av * (A' * X * B')) * Q;
denom = av * (Dleft(:) * Dright(:)') + lambda_z;
denom = max(denom, eps);
M = C ./ denom;
Zl = real(P * M * Q');
end

function [L, usedHyper, sigma] = build_hypergraph_laplacian(X, param)
% Build the manuscript's unnormalized hypergraph Laplacian using
% the hyperedges returned by gsp_nn_hypergraph.
    if exist('gsp_nn_hypergraph','file') ~= 2
        error('FedDFHMC:GSPBoxRequired', ...
            ['GSPBox function gsp_nn_hypergraph is required to construct ', ...
             'the hypergraph Laplacian. Install GSPBox and add it to the MATLAB path.']);
    end

    hgparam = struct();
    hgparam.k = param.hg_k;
    hgparam.rescale = 0;
    % Allow users to pass additional hypergraph params as param.hg_param (struct)
    if isfield(param,'hg_param') && isstruct(param.hg_param)
        fn = fieldnames(param.hg_param);
        for i = 1:numel(fn)
            hgparam.(fn{i}) = param.hg_param.(fn{i});
        end
    end
    try
        HG = gsp_nn_hypergraph(X', hgparam);
    catch err
        error('FedDFHMC:HypergraphConstructionFailed', ...
            'gsp_nn_hypergraph failed to construct the hypergraph: %s', err.message);
    end
    points = X';
    nEdges = numel(HG.E);
    edgeD2 = cell(nEdges,1);
    nDistances = sum(cellfun(@numel, HG.E));
    allDistances = zeros(nDistances,1);
    offset = 0;
    for e = 1:nEdges
        members = HG.E{e}(:);
        d2 = sum((points(members,:) - points(e,:)).^2, 2);
        edgeD2{e} = d2;
        count = numel(d2);
        allDistances(offset+(1:count)) = sqrt(d2);
        offset = offset + count;
    end

    if ~isempty(param.hg_sigma)
        sigma = param.hg_sigma;
    elseif isfield(hgparam, 'sigma') && ~isempty(hgparam.sigma)
        sigma = hgparam.sigma;
    else
        sigma = param.hg_sigma_scale * mean(allDistances)^2;
    end
    if ~isscalar(sigma) || ~isfinite(sigma) || sigma <= eps
        error('FedDFHMC:InvalidHypergraphSigma', ...
            'Hypergraph sigma must be a positive finite scalar.');
    end

    incidenceRows = zeros(nDistances,1);
    incidenceCols = zeros(nDistances,1);
    incidenceValues = zeros(nDistances,1);
    offset = 0;
    for e = 1:nEdges
        members = HG.E{e}(:);
        edgeWeight = sum(exp(-edgeD2{e}/sigma));
        count = numel(members);
        range = offset+(1:count);
        incidenceRows(range) = members;
        incidenceCols(range) = e;
        incidenceValues(range) = sqrt(edgeWeight);
        offset = offset + count;
    end
    weightedIncidence = sparse(incidenceRows, incidenceCols, incidenceValues, ...
        size(points,1), nEdges);

    % weightedIncidence = M*sqrt(W_s). Reconstruct Eq. (1):
    % L = D_v - M*W_s*D_e^(-1)*M'.
    edgeDegree = full(sum(weightedIncidence ~= 0, 1))';
    edgeDegree = max(edgeDegree, eps);
    vertexDegree = full(sum(weightedIncidence.^2, 2));
    invEdgeDegree = spdiags(1./edgeDegree, 0, numel(edgeDegree), numel(edgeDegree));
    Lsub = spdiags(vertexDegree, 0, numel(vertexDegree), numel(vertexDegree)) ...
        - weightedIncidence*invEdgeDegree*weightedIncidence';

    % Robustify: symmetrize (numerical) and sparsify
    Lsub = (Lsub + Lsub')/2;
    if ~issparse(Lsub)
        Lsub = sparse(Lsub);
    end
L = Lsub;
usedHyper = true;
end



function X = svt(M, t)
% Singular Value Thresholding: prox_{t||.||_*}(M)
% This is the closed-form global H update after the server obtains Hbar.
[U,S,V] = svd(M,'econ');
s = diag(S);
s = max(s - t, 0);
X = U*diag(s)*V';
end
