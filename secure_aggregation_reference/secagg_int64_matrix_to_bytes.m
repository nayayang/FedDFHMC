% Serialize int64 matrices for AEAD payloads.

function bytes = secagg_int64_matrix_to_bytes(values)
bytes = typecast(int64(values(:).'), 'uint8'); 
end
