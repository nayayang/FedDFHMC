% Serialize double matrices for AEAD downlink payloads.

function bytes = secagg_double_matrix_to_bytes(values)
bytes = typecast(double(values(:).'), 'uint8'); 
end
