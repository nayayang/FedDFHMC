% Client module: authenticate and decrypt the server downlink broadcast.

function [H, alpha] = secagg_client_receive_downlink(clientState, downlinkPacket)
% secagg_client_receive_downlink decrypts AEAD-protected (H, alpha).
%
% The client rejects packets for the wrong client, wrong channel, or tampered
% AEAD metadata/ciphertext.

if ~isfield(clientState, 'role') || ~strcmp(clientState.role, 'client')
    error('secagg_client_receive_downlink:InvalidRole', 'clientState must be a client state.');
end

if ~strcmp(downlinkPacket.channel, 'server-client-downlink') || ...
        downlinkPacket.clientId ~= clientState.clientId
    error('secagg_client_receive_downlink:WrongRecipient', ...
        'Downlink packet does not belong to this client.');
end

plaintext = AEADDec(clientState.downlinkKey, downlinkPacket.ciphertext, ...
    downlinkPacket.nonce, downlinkPacket.meta);

hByteCount = 8 * downlinkPacket.hRows * downlinkPacket.hCols; 
hBytes = plaintext(1:hByteCount); % Serialized H section.
alphaBytes = plaintext(hByteCount+1:end); % Serialized alpha section.

H = secagg_bytes_to_double_matrix(hBytes, downlinkPacket.hRows, downlinkPacket.hCols);
alpha = secagg_bytes_to_double_matrix(alphaBytes, downlinkPacket.alphaLength, 1);
end
