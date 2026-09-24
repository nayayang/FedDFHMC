% Client module: decrypt own TPA mask, mask Q^(v), and AEAD-encrypt upload.

function uploadPacket = secagg_client_create_upload(clientState, Qv, maskPacket, modulus, scale, clipThreshold, roundId)
% secagg_client_create_upload creates the protected upload C^(v).
%
% The client sees only its own Q^(v), its own Delta^(v), and its own keys.
% It never receives the aggregate mask or other clients' masks.

if ~isfield(clientState, 'role') || ~strcmp(clientState.role, 'client')
    error('secagg_client_create_upload:InvalidRole', 'clientState must be a client state.');
end

if maskPacket.clientId ~= clientState.clientId || maskPacket.roundId ~= roundId
    error('secagg_client_create_upload:WrongMaskPacket', ...
        'The encrypted mask packet does not belong to this client and round.');
end

[rows, cols] = size(Qv); % Q^(v) upload size.

if maskPacket.rows ~= rows || maskPacket.cols ~= cols
    error('secagg_client_create_upload:MaskShapeMismatch', ...
        'TPA mask dimensions do not match Qv.');
end

maskBytes = AEADDec(clientState.tpaMaskKey, maskPacket.ciphertext, ...
    maskPacket.nonce, maskPacket.meta); 
clientMask = secagg_bytes_to_int64_matrix(maskBytes, rows, cols); % Delta^(v).

encodedQ = secagg_encode_to_mod_ring(Qv, modulus, scale, clipThreshold);

maskedMessage = mod(encodedQ + clientMask, int64(modulus)); % M^(v)=Qhat^(v)+Delta^(v).

meta = sprintf(['channel=client-server-uplink;epoch=%d;round=%d;', ...
    'client=%d;rows=%d;cols=%d;modulus=%s;scale=%.17g'], ...
    clientState.keyEpoch, roundId, clientState.clientId, rows, cols, ...
    num2str(double(modulus)), scale);

nonce = secagg_make_nonce(roundId, clientState.clientId, 3); % Unique uplink nonce.
plaintext = secagg_int64_matrix_to_bytes(maskedMessage); % Serialize M^(v).
ciphertext = AEADEnc(clientState.uplinkKey, plaintext, nonce, meta); % C^(v).

uploadPacket = struct(); 
uploadPacket.channel = 'client-server-uplink'; 
uploadPacket.clientId = clientState.clientId; 
uploadPacket.roundId = roundId; 
uploadPacket.rows = rows; 
uploadPacket.cols = cols; 
uploadPacket.modulus = int64(modulus); 
uploadPacket.scale = scale;
uploadPacket.nonce = nonce; 
uploadPacket.meta = meta; 
uploadPacket.ciphertext = ciphertext; 
end
