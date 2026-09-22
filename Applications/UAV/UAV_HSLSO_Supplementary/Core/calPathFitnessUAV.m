function fitness = calPathFitnessUAV(startPos, goalPos, X, Y, Z, path, P)
    %% ---------- 1. 路径裁剪 ----------
    path(:,1:2) = max(min(path(:,1:2), max(X(:))), min(X(:)));
    path(:,3)   = max(min(path(:,3), P.z_max), P.z_min);

    fullPath = [startPos; path; goalPos];
    nSeg = size(fullPath, 1) - 1;

    %% ---------- 2. 路径长度 ----------
    diffP = diff(fullPath,1,1);
    L = sum(vecnorm(diffP,2,2));

    %% ---------- 3. 地形碰撞与净空检查（向量化） ----------
    % 构建每段的采样点
    t = linspace(0,1,P.n_samples_seg)';        % [n_samples_seg x 1]
    seg_vectors = reshape(fullPath(2:end,:) - fullPath(1:end-1,:), [1,nSeg,3]); % [1 x nSeg x 3]
    seg_start   = reshape(fullPath(1:end-1,:), [1,nSeg,3]);                      % [1 x nSeg x 3]

    seg_samples = seg_start + t .* seg_vectors; % [n_samples_seg x nSeg x 3]

    % 插值地形高度
    h = interp2(X, Y, Z, seg_samples(:,:,1), seg_samples(:,:,2), 'linear', 0);
    clearance = seg_samples(:,:,3) - h;

    % 碰撞检查
    if any(clearance(:) < 0)
        fitness = P.big_collision + L;
        return;
    end

    min_clear = min(clearance(:));
    n_violate = sum(clearance(:) < P.min_clearance);

    %% ---------- 4. 转角约束（向量化） ----------
    soft_pen = 0;
    if nSeg >= 2
        v1 = diffP(1:end-1,:); % [nSeg-1 x 3]
        v2 = diffP(2:end,:);   % [nSeg-1 x 3]

        v1n = v1 ./ (vecnorm(v1,2,2)+eps);
        v2n = v2 ./ (vecnorm(v2,2,2)+eps);

        cosang = sum(v1n .* v2n, 2);
        cosang = max(min(cosang,1),-1);
        ang = acosd(cosang);

        over_turn = max(0, ang - P.max_turn_deg);
        soft_pen = soft_pen + P.soft_penalty * sum(over_turn.^2);
    end

    %% ---------- 5. 单段爬升/下降约束 ----------
    climb = diffP(:,3);
    over_climb = max(0, abs(climb) - P.max_climb_per_seg);
    soft_pen = soft_pen + P.soft_penalty * sum(over_climb.^2);

    %% ---------- 6. 净空不足惩罚 ----------
    if min_clear < P.min_clearance
        soft_pen = soft_pen + P.soft_penalty * (P.min_clearance - min_clear)^2;
    end

    %% ---------- 7. 总适应度 ----------
    fitness = L + soft_pen;
    if ~isfinite(fitness)
        fitness = P.big_collision;
    end
end
