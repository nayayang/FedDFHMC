% AES-GCM encryption used by the FedDFHMC secure aggregation protocol.
function ciphertext = AEADEnc(key, plaintext, nonce, meta)
% AEADEnc implements AEAD.Enc(K, message; nonce, meta) with AES-GCM.
%
% Inputs:
%   key       : 16, 24, or 32 uint8 bytes for AES-128/192/256.
%   plaintext : uint8 byte vector.
%   nonce     : unique uint8 byte vector. AES-GCM normally uses 12 bytes.
%   meta      : associated data authenticated but not encrypted.
%
% Output:
%   ciphertext: uint8 vector containing encrypted bytes followed by GCM tag.

if ~usejava('jvm')
    error('AEADEnc:NoJVM', 'AES-GCM AEAD requires MATLAB Java support.');
end

validate_aead_inputs(key, nonce);

import javax.crypto.Cipher
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.SecretKeySpec

cipher = Cipher.getInstance('AES/GCM/NoPadding');
keySpec = SecretKeySpec(to_java_bytes(key), 'AES');
gcmSpec = GCMParameterSpec(128, to_java_bytes(nonce));
cipher.init(Cipher.ENCRYPT_MODE, keySpec, gcmSpec);

% Associated data is authenticated but not encrypted.
cipher.updateAAD(to_java_bytes(meta_to_bytes(meta)));

encrypted = cipher.doFinal(to_java_bytes(plaintext));
ciphertext = from_java_bytes(encrypted);
end

function validate_aead_inputs(key, nonce)
keyLength = numel(key);
if ~any(keyLength == [16, 24, 32])
    error('AEADEnc:InvalidKeyLength', ...
        'AES key must contain 16, 24, or 32 bytes.');
end
if isempty(nonce)
    error('AEADEnc:InvalidNonce', 'Nonce must be nonempty and unique.');
end
end

function bytes = meta_to_bytes(meta)
if isstring(meta) || ischar(meta)
    bytes = unicode2native(char(meta), 'UTF-8');
else
    bytes = uint8(meta);
end
end

function jbytes = to_java_bytes(bytes)
jbytes = typecast(uint8(bytes(:).'), 'int8');
end

function bytes = from_java_bytes(jbytes)
bytes = typecast(int8(jbytes(:).'), 'uint8');
end
