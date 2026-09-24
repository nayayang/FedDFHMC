% TPA module: generate fresh masks and distribute encrypted mask material.

function [maskPackets, serverMaskPacket] = secagg_tpa_prepare_round(tpaState, numClients, matrixSize, modulus, roundId)
% secagg_tpa_prepare_round prepares one round of TPA-assisted masking.
%
% The TPA sends Delta^(v) only to client v and sends Delta_Sigma only to the
% server. Individual masks never leave this function except inside encrypted
% TPA-to-client packets.

if ~isfield(tpaState, 'role') || ~strcmp(tpaState.role, 'tpa')
    error('secagg_tpa_prepare_round:InvalidRole', 'tpaState must be a TPA state.');
end

rows = matrixSize(1); 
cols = matrixSize(2); 

modulus = int64(modulus);

maskPackets = cell(numClients, 1); % Encrypted Delta^(v) for each client.

aggregateMask = zeros(rows, cols, 'int64'); 

for v = 1:numClients
    clientMask = secagg_random_mask(rows, cols, modulus); % Fresh one-time additive mask.
    aggregateMask = mod(aggregateMask + clientMask, modulus); 

    meta = sprintf(['channel=tpa-client-mask;epoch=%d;round=%d;', ...
        'client=%d;rows=%d;cols=%d;modulus=%s'], ...
        tpaState.keyEpoch, roundId, v, rows, cols, num2str(double(modulus)));
    nonce = secagg_make_nonce(roundId, v, 1);
    plaintext = secagg_int64_matrix_to_bytes(clientMask); 
    ciphertext = AEADEnc(tpaState.clientMaskKeys{v}, plaintext, nonce, meta); 

    packet = struct(); 
    packet.channel = 'tpa-client-mask'; % Authenticated channel label.
    packet.clientId = v;
    packet.roundId = roundId; 
    packet.rows = rows;
    packet.cols = cols; 
    packet.modulus = modulus; % Ring modulus.
    packet.nonce = nonce; 
    packet.meta = meta; 
    packet.ciphertext = ciphertext; % Encrypted individual mask.
    maskPackets{v} = packet; 
end

serverMeta = sprintf(['channel=tpa-server-aggregate-mask;epoch=%d;', ...
    'round=%d;client=0;rows=%d;cols=%d;modulus=%s'], ...
    tpaState.keyEpoch, roundId, rows, cols, num2str(double(modulus)));
serverNonce = secagg_make_nonce(roundId, 0, 2); % Unique AEAD nonce for TPA->server.
serverPlaintext = secagg_int64_matrix_to_bytes(aggregateMask); % Serialize Delta_Sigma.
serverCiphertext = AEADEnc(tpaState.serverAggregateKey, serverPlaintext, serverNonce, serverMeta);

serverMaskPacket = struct(); 
serverMaskPacket.channel = 'tpa-server-aggregate-mask'; 
serverMaskPacket.clientId = 0; 
serverMaskPacket.roundId = roundId; 
serverMaskPacket.rows = rows; 
serverMaskPacket.cols = cols; 
serverMaskPacket.modulus = modulus; 
serverMaskPacket.nonce = serverNonce; 
serverMaskPacket.meta = serverMeta; 
serverMaskPacket.ciphertext = serverCiphertext; 

end
