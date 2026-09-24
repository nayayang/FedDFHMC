% AES-GCM decryption used by the FedDFHMC secure aggregation protocol.
function plaintext = AEADDec(key, ciphertext, nonce, meta)
% AEADDec implements AEAD.Dec(K, ciphertext; nonce, meta) with AES-GCM.
% Decryption fails if ciphertext, tag, nonce, key, or associated metadata
% has been modified.

if ~usejava('jvm')
    error('AEADDec:NoJVM', 'AES-GCM AEAD requires MATLAB Java support.');
end

% The same key and nonce used by AEADEnc are required for decryption.
validate_aead_inputs(key, nonce);

import javax.crypto.Cipher
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.SecretKeySpec

% Java expects the GCM tag to be appended to the ciphertext input.
cipher = Cipher.getInstance('AES/GCM/NoPadding');
keySpec = SecretKeySpec(to_java_bytes(key), 'AES');
gcmSpec = GCMParameterSpec(128, to_java_bytes(nonce));
cipher.init(Cipher.DECRYPT_MODE, keySpec, gcmSpec);

% AAD must match encryption exactly. Any change causes authentication fail.
cipher.updateAAD(to_java_bytes(meta_to_bytes(meta)));

try
    decrypted = cipher.doFinal(to_java_bytes(ciphertext));
catch err

    error('AEADDec:AuthenticationFailed', ...
        'AEAD authentication failed: %s', err.message);
end
plaintext = from_java_bytes(decrypted);
end

function validate_aead_inputs(key, nonce)
% Validate only properties needed by AES-GCM construction.
keyLength = numel(key);
if ~any(keyLength == [16, 24, 32])
    error('AEADDec:InvalidKeyLength', ...
        'AES key must contain 16, 24, or 32 bytes.');
end
if isempty(nonce)
    error('AEADDec:InvalidNonce', 'Nonce must be nonempty and unique.');
end
end

function bytes = meta_to_bytes(meta)
% Convert MATLAB char/string metadata to UTF-8 bytes for AAD.
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
