% Cryptographic random mask sampling over Z_R.

function mask = secagg_random_mask(rows, cols, modulus)
% secagg_random_mask samples an int64 mask matrix from the ring Z_modulus.

modulus = int64(modulus); % Ring modulus R.

if modulus <= 1 || double(modulus) > 2^32
    error('secagg_random_mask:UnsupportedModulus', ...
        'This sampler expects 1 < modulus <= 2^32.');
end

targetCount = rows * cols; 
base = 2^32; 
modulusDouble = double(modulus); % Modulus as double for rejection math.
limit = floor(base / modulusDouble) * modulusDouble; % Largest unbiased range.

values = zeros(targetCount, 1, 'int64'); % Output vector before reshape.
filled = 0; 
while filled < targetCount
    batchCount = max(16, ceil(1.2 * (targetCount - filled))); 
    randomBytes = aead_random_bytes(4 * batchCount); 
    candidates = double(typecast(uint8(randomBytes), 'uint32')); 
    candidates = candidates(candidates < limit); % Reject biased tail.
    accepted = int64(mod(candidates, modulusDouble)); % Map to Z_R.
    take = min(numel(accepted), targetCount - filled); 
    values(filled+1:filled+take) = accepted(1:take);
    filled = filled + take;
end

mask = reshape(values, rows, cols); % Convert vector to rows-by-cols mask.
end
