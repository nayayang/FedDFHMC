% Fixed-point encoding into the modular ring Z_R.

function encoded = secagg_encode_to_mod_ring(values, modulus, scale, clipThreshold)
% Quant_s,c(values) = round(s * clip(values, -c, c)) mod R.

if ~isinf(clipThreshold)
    values = min(max(values, -clipThreshold), clipThreshold); 
end

fixedPoint = int64(round(values * scale)); 

encoded = mod(fixedPoint, int64(modulus)); 
end
