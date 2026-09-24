% Deserialize int64 matrices from AEAD payloads.

function values = secagg_bytes_to_int64_matrix(bytes, rows, cols)
values = typecast(uint8(bytes(:).'), 'int64'); 
values = reshape(values, rows, cols); 
end
