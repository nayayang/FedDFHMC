function G = gsp_create_laplacian(G, ~)
%GSP_CREATE_LAPLACIAN Create the normalized hypergraph Laplacian.
% This trimmed GSPBox copy retains the branch used by gsp_nn_hypergraph.
%
% Copyright (C) 2013-2016 Nathanael Perraudin, Johan Paratte,
% David I Shuman. Distributed under GPLv3+; see ../LICENSE.

if numel(G) > 1
    for i = 1:numel(G)
        G{i} = gsp_create_laplacian(G{i});
    end
    return;
end
if ~isfield(G, 'hypergraph') || ~G.hypergraph
    error('gsp_create_laplacian:UnsupportedGraphType', ...
        'This repository includes only the hypergraph Laplacian branch.');
end

G.de = sum(G.W > 0, 1)';
G.dv = sum(G.W.^2, 2);
G.A = G.W*G.W' - diag(G.dv);
G.L = eye(G.N) - diag(G.dv.^(-0.5)) * G.W * ...
    diag(G.de.^(-1)) * G.W' * diag(G.dv.^(-0.5));
G.lap_type = 'normalized';
end
