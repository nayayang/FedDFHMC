function D = gsp_distanz(X, Y, P)
%GSP_DISTANZ Calculate distances between all column vectors in X and Y.
% This file is from GSPBox version 0.7.5 and is distributed under GPLv3+.
%
% Copyright (C) 2013-2016 Nathanael Perraudin, Johan Paratte,
% David I Shuman.

if nargin < 1
    error('Not enough input parameters.');
end
if nargin < 2
    Y = X;
end
[rx, cx] = size(X);
[ry, cy] = size(Y);
if rx ~= ry
    error('The sizes of X and Y do not fit.');
end
if nargin < 3
    xx = sum(X.*X, 1);
    yy = sum(Y.*Y, 1);
    xy = X'*Y;
    D = abs(repmat(xx', [1 cy]) + repmat(yy, [cx 1]) - 2*xy);
else
    [rp, rp2] = size(P);
    if rx ~= rp
        error('The sizes of X and P do not fit.');
    end
    if rp2 ~= rp
        error('P must be square.');
    end
    xx = sum(X .* (P*X), 1);
    yy = sum(Y .* (P*Y), 1);
    xy = X'*(P*Y);
    yx = Y'*(P*X);
    D = abs(repmat(xx', [1 cy]) + repmat(yy, [cx 1]) - xy - yx);
end
D = sqrt(D);
if nargin < 2
    D(1:cx+1:end) = 0;
end
end
