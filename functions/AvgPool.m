function y = AvgPool(x,S)
 
    [xrow, xcol, numFilters] = size(x);

    y = zeros(xrow/S, xcol/S, numFilters);
    for k1 = 1:numFilters
        filter = ones(S) / (S*S);    % for mean
        image  = conv2(x(:, :, k1), filter, 'valid');

        y(:, :, k1) = image(1:S:end, 1:S:end);
    end

end