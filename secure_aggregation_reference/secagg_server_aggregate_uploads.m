% Server module: decrypt masked uploads and recover only the aggregate.

function Qbar = secagg_server_aggregate_uploads(serverState, uploadPackets, serverMaskPacket, modulus, scale, matrixSize, roundId)
% secagg_server_aggregate_uploads implements the server side of Sec. 5.
%
% The server decrypts M^(v), sums masked messages, decrypts Delta_Sigma from
% the TPA, removes only Delta_Sigma, and decodes the aggregate average.

if ~isfield(serverState, 'role') || ~strcmp(serverState.role, 'server')
    error('secagg_server_aggregate_uploads:InvalidRole', 'serverState must be a server state.');
end

rows = matrixSize(1); 
cols = matrixSize(2); 
V = numel(uploadPackets); 
modulus = int64(modulus); 

if V ~= serverState.numClients
    error('secagg_server_aggregate_uploads:ClientCountMismatch', ...
        'The number of upload packets does not match serverState.numClients.');
end

secagg_assert_packet(serverMaskPacket, 'tpa-server-aggregate-mask', 0, roundId, rows, cols, modulus);
aggregateMaskBytes = AEADDec(serverState.tpaAggregateKey, ...
    serverMaskPacket.ciphertext, serverMaskPacket.nonce, serverMaskPacket.meta);
aggregateMask = secagg_bytes_to_int64_matrix(aggregateMaskBytes, rows, cols); 

maskedSum = zeros(rows, cols, 'int64'); % S_masked accumulator.
seenClient = false(1, V); 
for i = 1:V
    packet = uploadPackets{i}; 
    clientId = packet.clientId; 
    secagg_assert_packet(packet, 'client-server-uplink', clientId, roundId, rows, cols, modulus);
    if clientId < 1 || clientId > V || seenClient(clientId)
        error('secagg_server_aggregate_uploads:DuplicateOrInvalidClient', ...
            'Each client must upload exactly once per round.');
    end
    seenClient(clientId) = true; 

    plaintext = AEADDec(serverState.uplinkKeys{clientId}, ...
        packet.ciphertext, packet.nonce, packet.meta); % Recover M^(v), not Q^(v).
    maskedMessage = secagg_bytes_to_int64_matrix(plaintext, rows, cols);
    maskedSum = mod(maskedSum + maskedMessage, modulus); % S_masked += M^(v).
end

encodedSum = mod(maskedSum - aggregateMask, modulus); % Sum_v Qhat^(v).

decodedSum = secagg_decode_from_mod_ring(encodedSum, modulus, scale); 
Qbar = decodedSum / V; % Average used by the global H update.

end

function secagg_assert_packet(packet, channel, clientId, roundId, rows, cols, modulus)
% Check non-secret packet fields before AEAD decryption.
if ~isfield(packet, 'channel') || ~strcmp(packet.channel, channel)
    error('secagg_server_aggregate_uploads:WrongChannel', 'Unexpected packet channel.');
end
if packet.clientId ~= clientId || packet.roundId ~= roundId
    error('secagg_server_aggregate_uploads:WrongRoundOrClient', ...
        'Unexpected client id or round id.');
end
if packet.rows ~= rows || packet.cols ~= cols || int64(packet.modulus) ~= int64(modulus)
    error('secagg_server_aggregate_uploads:ShapeOrModulusMismatch', ...
        'Unexpected packet dimensions or modulus.');
end
end
