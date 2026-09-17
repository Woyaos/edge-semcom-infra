function model = load_learned_constellation16(mat_path)
% Load learned constellation exported by train_learned_16qam_matlab_ofdm.py.

if nargin < 1
    mat_path = 'trained_constellation_16qam_ofdm.mat';
end

s = load(mat_path);

% Support both legacy exports and train_constellation_fresh.py exports.
if isfield(s, 'constellation')
    c_raw = s.constellation;
elseif isfield(s, 'constellation_best')
    c_raw = s.constellation_best;
elseif isfield(s, 'real') && isfield(s, 'imag')
    c_raw = complex(s.real, s.imag);
else
    error('Missing constellation/constellation_best in %s', mat_path);
end

if isfield(s, 'M')
    M = double(s.M(1));
elseif isfield(s, 'config_M')
    M = double(s.config_M(1));
else
    error('Missing field M/config_M in %s', mat_path);
end

if isfield(s, 'k')
    k = double(s.k(1));
elseif isfield(s, 'config_k')
    k = double(s.config_k(1));
else
    k = round(log2(M));
end

if ~isfield(s, 'bit_table')
    error('Missing field bit_table in %s', mat_path);
end

model = struct();
if ~isreal(c_raw)
    model.constellation = c_raw(:);
elseif ismatrix(c_raw) && size(c_raw, 2) == 2
    model.constellation = complex(c_raw(:, 1), c_raw(:, 2));
elseif ismatrix(c_raw) && size(c_raw, 1) == 2
    model.constellation = complex(c_raw(1, :).', c_raw(2, :).');
else
    error('Unsupported constellation format in %s', mat_path);
end

model.bit_table = uint8(s.bit_table);
model.M = M;
model.k = k;

if size(model.bit_table, 1) ~= model.M || size(model.bit_table, 2) ~= model.k
    error('bit_table shape mismatch, expected [%d,%d]', model.M, model.k);
end

if numel(model.constellation) ~= model.M
    error('constellation size mismatch, expected %d points', model.M);
end
end

