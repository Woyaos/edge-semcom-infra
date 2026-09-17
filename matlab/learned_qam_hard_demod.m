function bits_out = learned_qam_hard_demod(rx_symbols, model, n_bits_out)
% Nearest-neighbor hard demod with learned constellation.
% rx_symbols: column vector [Nsym x 1]
% bits_out  : row vector [1 x n_bits_out]

rx = rx_symbols(:);
const = model.constellation(:).';

% Distance matrix: [Nsym, M]
d2 = abs(rx - const).^2;
[~, idx_hat] = min(d2, [], 2);

bits_hat = model.bit_table(idx_hat, :);
bits_col = reshape(bits_hat.', [], 1);

if nargin < 3 || isempty(n_bits_out)
    n_bits_out = numel(bits_col);
end

n_bits_out = min(double(n_bits_out), numel(bits_col));
bits_col = bits_col(1:n_bits_out);

bits_out = double(bits_col).';
end

