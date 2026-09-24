% Decode fixed-point values from the modular ring Z_R.

function values = secagg_decode_from_mod_ring(encoded, modulus, scale)
% secagg_decode_from_mod_ring converts Z_R elements back to real values.

modulus = int64(modulus); 
halfModulus = int64(floor(double(modulus) / 2)); 
encoded = int64(encoded); 
fixedPoint = encoded; 
negative = encoded > halfModulus; % Entries representing negative values.
fixedPoint(negative) = encoded(negative) - modulus; % Recover signed integers.

values = double(fixedPoint) / scale; 
end
