function context = secagg_key_manager(numClients)
% secagg_key_manager creates independent AEAD keys for all secure channels.
%
% Channels:
%   client -> server : uplink masked Q^(v)
%   server -> client : downlink broadcast (H, alpha)
%   TPA    -> client : individual additive mask Delta^(v)
%   TPA    -> server : aggregate mask Delta_Sigma

if nargin < 1 || isempty(numClients) || numClients < 1
    error('secagg_key_manager:InvalidClientCount', ...
        'numClients must be a positive integer.');
end

clients = cell(numClients, 1); 
server = struct(); 
tpa = struct(); 

server.role = 'server'; 
server.keyEpoch = 1; 
server.numClients = numClients; 
server.uplinkKeys = cell(numClients, 1); 
server.downlinkKeys = cell(numClients, 1); 

tpa.role = 'tpa'; 
tpa.keyEpoch = 1; 
tpa.numClients = numClients; 
tpa.clientMaskKeys = cell(numClients, 1); 

tpaServerKey = aead_random_bytes(32); % AES-256 key shared only by TPA and server.
server.tpaAggregateKey = tpaServerKey; 
tpa.serverAggregateKey = tpaServerKey; 

for v = 1:numClients
    uploadKey = aead_random_bytes(32); 
    downlinkKey = aead_random_bytes(32); 
    maskKey = aead_random_bytes(32); 

    server.uplinkKeys{v} = uploadKey; % Server receives this client's upload.
    server.downlinkKeys{v} = downlinkKey; 
    tpa.clientMaskKeys{v} = maskKey; % TPA encrypts this client's mask.

    client = struct(); 
    client.role = 'client'; 
    client.clientId = v; 
    client.keyEpoch = 1; 
    client.uplinkKey = uploadKey; 
    client.downlinkKey = downlinkKey; 
    client.tpaMaskKey = maskKey; 
    clients{v} = client; % Save role-specific client state.
end

context = struct(); 
context.server = server; 
context.tpa = tpa; 
context.clients = clients; % Pass one client sub-struct to each client.
end
