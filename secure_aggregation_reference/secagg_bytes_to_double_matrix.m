% Deserialize double matrices from AEAD downlink payloads.

function values = secagg_bytes_to_double_matrix(bytes, rows, cols)
values = typecast(uint8(bytes(:).'), 'double'); 
values = reshape(values, rows, cols); 
end
