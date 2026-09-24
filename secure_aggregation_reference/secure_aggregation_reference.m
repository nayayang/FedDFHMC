function Qbar = secure_aggregation_reference(Hlocal, dualVars, modulus, scale, clipThreshold, roundId, securityContext)
% secure_aggregation_reference runs one complete secure upload aggregation.
%
%   1. TPA module generates fresh masks and sends only aggregate mask to server.
%   2. Client module receives its own mask, masks Q^(v), and AEAD-encrypts upload.
%   3. Server module decrypts uploads, removes only aggregate mask, and decodes sum.
%
% Secrets are passed only to the role-specific module that owns them.

if nargin < 3 || isempty(modulus)
    modulus = int64(2147483647); % Default ring modulus R.
end

if nargin < 4 || isempty(scale)
    scale = 1e4; % Default quantization scale s.
end

V = numel(Hlocal); 

if V == 0
    error('secure_aggregation_reference:EmptyClients', ...
        'Hlocal must contain at least one client matrix.');
end

if numel(dualVars) ~= V
    error('secure_aggregation_reference:DualCountMismatch', ...
        'dualVars must have the same number of cells as Hlocal.');
end

modulus = int64(modulus); 

[k, n] = size(Hlocal{1}); % Q^(v) has k rows and n columns.

for v = 1:V
    if ~isequal(size(Hlocal{v}), [k, n]) || ~isequal(size(dualVars{v}), [k, n])
        error('secure_aggregation_reference:ShapeMismatch', ...
            'All Hlocal and dualVars matrices must have the same size.');
    end
end

if nargin < 5 || isempty(clipThreshold)
    clipThreshold = floor((double(modulus) - 1) / (2 * V * scale));
end

if nargin < 6 || isempty(roundId)
    roundId = 1; 
end

if nargin < 7 || isempty(securityContext)
    securityContext = secagg_key_manager(V); % Fresh role-separated keys.
end

secagg_validate_context(securityContext, V);

[maskPackets, serverMaskPacket] = secagg_tpa_prepare_round( ...
    securityContext.tpa, V, [k, n], modulus, roundId);

uploadPackets = cell(V, 1); % Each element is one client-to-server ciphertext.
for v = 1:V
    Qv = Hlocal{v} + dualVars{v}; % Q^(v)=H^(v)+U^(v) in the paper.
    uploadPackets{v} = secagg_client_create_upload( ...
        securityContext.clients{v}, Qv, maskPackets{v}, ...
        modulus, scale, clipThreshold, roundId);
end

Qbar = secagg_server_aggregate_uploads( ...
    securityContext.server, uploadPackets, serverMaskPacket, ...
    modulus, scale, [k, n], roundId);

end

function secagg_validate_context(securityContext, V)
% Ensure each role-specific sub-struct exists before protocol execution.
requiredFields = {'server', 'tpa', 'clients'}; 
for i = 1:numel(requiredFields)
    if ~isfield(securityContext, requiredFields{i})
        error('secure_aggregation_reference:InvalidContext', ...
            'securityContext.%s is missing.', requiredFields{i});
    end
end
if numel(securityContext.clients) ~= V
    error('secure_aggregation_reference:ClientContextMismatch', ...
        'securityContext.clients must contain one state per client.');
end
end
