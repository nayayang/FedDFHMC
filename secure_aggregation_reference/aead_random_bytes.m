% Cryptographically secure random-byte generator for FedDFHMC.
function bytes = aead_random_bytes(n)
% Return n random bytes for AEAD keys or nonce random suffixes.

if nargin < 1 || n < 1 || n ~= floor(n)
    error('aead_random_bytes:InvalidLength', ...
        'n must be a positive integer.');
end

if usejava('jvm')
    try
        % Prefer Java SecureRandom for cryptographic key/nonce material.
        rngObj = javaObject('java.security.SecureRandom');
        raw = rngObj.generateSeed(n);
        bytes = reshape(typecast(int8(raw), 'uint8'), 1, []);
        if numel(bytes) ~= n
            error('aead_random_bytes:JavaLengthMismatch', ...
                'Java SecureRandom returned an unexpected byte count.');
        end
        return;
    catch
        % Fall through to MATLAB RNG below.
    end
end

% Fallback keeps tests runnable without Java, but is not cryptographic.
bytes = uint8(randi([0, 255], 1, n));
end
