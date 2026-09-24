% AEAD nonce construction for all secure aggregation channels.

function nonce = secagg_make_nonce(roundId, partyId, directionCode)
% secagg_make_nonce builds a 96-bit AES-GCM nonce.

roundBytes = typecast(uint32(roundId), 'uint8');
partyBytes = typecast(uint16(partyId), 'uint8'); % Bind nonce to client/server id.
directionBytes = typecast(uint16(directionCode), 'uint8'); % Separate channel directions.

randomBytes = aead_random_bytes(4); 

nonce = [roundBytes(:); partyBytes(:); directionBytes(:); randomBytes(:)].';
end
