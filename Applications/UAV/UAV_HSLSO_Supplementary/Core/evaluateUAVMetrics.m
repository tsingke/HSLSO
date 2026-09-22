function M = evaluateUAVMetrics(path, startPos, goalPos, X, Y, Z, P)
    path(:,1:2) = max(min(path(:,1:2), max(X(:))), min(X(:)));
    path(:,3)   = max(min(path(:,3), P.z_max), P.z_min);

    fullPath = [startPos; path; goalPos];
    diffP = diff(fullPath, 1, 1);

    M.Lp = sum(vecnorm(diffP, 2, 2));

    %% 平均转角 / 最大转角
    if size(fullPath,1) >= 3
        v1 = diffP(1:end-1,:);
        v2 = diffP(2:end,:);

        v1n = v1 ./ (vecnorm(v1,2,2) + eps);
        v2n = v2 ./ (vecnorm(v2,2,2) + eps);

        cosang = sum(v1n .* v2n, 2);
        cosang = max(min(cosang, 1), -1);

        ang = acosd(cosang);

        M.kappa_avg = mean(ang);
        M.kappa_max = max(ang);
    else
        M.kappa_avg = NaN;
        M.kappa_max = NaN;
    end

    %% 爬升
    climb = abs(diffP(:,3));
    M.climb_avg = mean(climb);
    M.climb_max = max(climb);

    %% 净空与违规率
    nSeg = size(fullPath,1) - 1;

    min_clear = inf;
    n_violate = 0;
    n_collision = 0;
    n_total = 0;

    for i = 1:nSeg
        seg = linspace(0,1,P.n_samples_seg)' .* ...
              (fullPath(i+1,:) - fullPath(i,:)) + fullPath(i,:);

        h = interp2(X, Y, Z, seg(:,1), seg(:,2), 'linear', 0);
        clearance = seg(:,3) - h;

        min_clear = min(min_clear, min(clearance));
        n_violate = n_violate + sum(clearance < P.min_clearance);
        n_collision = n_collision + sum(clearance < 0);
        n_total = n_total + numel(clearance);
    end

    M.Cmin = min_clear;
    M.Vr = 100 * n_violate / max(1, n_total);
    M.Collision = 100 * n_collision / max(1, n_total);
end
