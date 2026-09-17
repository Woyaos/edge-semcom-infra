function symbols = learned_qam_mod(bits_in, model)
% Hard lookup modulation using learned constellation and bit_table.
% bits_in: column vector [Nbits x 1], values 0/1
% symbols: column vector [Nsym x 1]

bits = uint8(bits_in(:));
if any(bits ~= 0 & bits ~= 1)
    error('bits_in must contain only 0/1');
end

k = model.k;
M = model.M;

if mod(numel(bits), k) ~= 0
    error('Number of bits must be a multiple of k=%d', k);
end

num_sym = numel(bits) / k;
bits_mat = reshape(bits, k, num_sym).';

weights = uint32(2.^((k-1):-1:0));
dec_idx = sum(uint32(bits_mat) .* weights, 2);

% bit_table is indexed by binary symbol index 0..M-1.
% Verify mapping consistency once to avoid silent mismatch.
if any(any(model.bit_table ~= uint8(dec2bin(0:M-1, k) - '0')))
    error('bit_table is not binary-indexed in expected order');
end

symbols = model.constellation(double(dec_idx) + 1);
end

