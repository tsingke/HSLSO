function path = decodePath(x)
    global UAV_CTX

    x = x(:).';
    path = reshape(x, 3, []).';

    if isempty(UAV_CTX)
        return;
    end

    X = UAV_CTX.X;
    P = UAV_CTX.params;

    path(:,1:2) = max(min(path(:,1:2), max(X(:))), min(X(:)));
    path(:,3)   = max(min(path(:,3), P.z_max), P.z_min);
end
