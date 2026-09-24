% Server module: AEAD-protect the downlink broadcast (H, alpha).

function downlinkPackets = secagg_server_create_downlink(serverState, H, alpha, roundId)
% secagg_server_create_downlink encrypts the global broadcast for each client.
%
% This implements the paper's downlink AEAD step:
% B^(v) = AEAD.Enc(K_v_down, (H, alpha); nonce_d, meta_d).

if ~isfield(serverState, 'role') || ~strcmp(serverState.role, 'server')
    error('secagg_server_create_downlink:InvalidRole', 'serverState must be a server state.');
end

[hRows, hCols] = size(H); 
alpha = alpha(:); 
alphaLength = numel(alpha); 
numClients = serverState.numClients; 

downlinkPackets = cell(numClients, 1); 

hBytes = secagg_double_matrix_to_bytes(H); 
alphaBytes = secagg_double_matrix_to_bytes(alpha); 
plaintext = [hBytes(:); alphaBytes(:)].'; 

for v = 1:numClients
    meta = sprintf(['channel=server-client-downlink;epoch=%d;round=%d;', ...
        'client=%d;hRows=%d;hCols=%d;alphaLength=%d'], ...
        serverState.keyEpoch, roundId, v, hRows, hCols, alphaLength);
    nonce = secagg_make_nonce(roundId, v, 4); % Unique downlink nonce.
    ciphertext = AEADEnc(serverState.downlinkKeys{v}, plaintext, nonce, meta);

    packet = struct(); 
    packet.channel = 'server-client-downlink'; % Channel label.
    packet.clientId = v; 
    packet.roundId = roundId; 
    packet.hRows = hRows; 
    packet.hCols = hCols; 
    packet.alphaLength = alphaLength; 
    packet.nonce = nonce; 
    packet.meta = meta; 
    packet.ciphertext = ciphertext; % Encrypted (H, alpha).
    downlinkPackets{v} = packet; 
end
