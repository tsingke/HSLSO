function [bestPath, gbesthistory] = runSLPSO(Fitness, popsize_in, dimension, xmax, xmin, maxiter)
    % 1. SLPSO 的自适应种群计算
    M = 100;
    m = M + floor(dimension / 10); % 传入的 popsize_in 不用于初始化，用 m 代替
    
    % 2. 统一框架的最大预算 (确保与其他算法公平对比)
    MaxFEs = popsize_in * maxiter; 
    FEs = 0;
    
    c3 = (dimension / M) * 0.01;
    PL = zeros(m, 1);
    for i = 1:m
        PL(i) = (1 - (i - 1) / m)^log(sqrt(ceil(dimension / M))); % 学习概率
    end
    
    % 初始化种群
     p = xmin + (xmax - xmin) .* rand(m, dimension);
    fitness = zeros(m, 1);
    for i = 1:m
        fitness(i) = Fitness(p(i, :)'); 
    end
    FEs = FEs + m;
    
     v = zeros(m, dimension);
    [bestever, best_idx] = min(fitness);
    gbestx = p(best_idx, :);
    
    % 初始化历史记录（按代数记录，对齐主框架作图）
    gbesthistory = zeros(1, maxiter);
    iter = 1;
    gbesthistory(iter) = bestever;
    
    %% 主循环
    while FEs < MaxFEs
        iter = iter + 1;
        if iter > maxiter; break; end % 防止画图索引越界的保护
        
        % 降序排序（差到好：排名 1 是最差，排名 m 是最好）
        [fitness, rank] = sort(fitness, 'descend');
        p = p(rank, :);
        v = v(rank, :);
        
        % 更新历史最优
        current_besty = fitness(m);
        current_bestp = p(m, :);
        if current_besty < bestever
            bestever = current_besty;
            gbestx = current_bestp;
        end
        
        center = mean(p); % 当前种群平均位置
        
        randco1 = rand(m, dimension);
        randco2 = rand(m, dimension);
        randco3 = rand(m, dimension);
        
        % 寻找学习对象（必须比自己排名高）
        winidxmask = repmat((1:m)', [1, dimension]);
        winidx = winidxmask + ceil(rand(m, dimension) .* (m - winidxmask));
        
        pwin = zeros(m, dimension);
        for j = 1:dimension
            pwin(:, j) = p(winidx(:, j), j);
        end
        
        % 对除了最好粒子外的其他粒子进行概率更新
        for i = 1:m-1
            if rand < PL(i)
                v(i, :) = randco1(i, :) .* v(i, :) + ...
                          randco2(i, :) .* (pwin(i, :) - p(i, :)) + ...
                          c3 * randco3(i, :) .* (center - p(i, :));
                p(i, :) = p(i, :) + v(i, :);
                
                % 边界约束
                p(i, :) = max(p(i, :), xmin);
                p(i, :) = min(p(i, :), xmax);
                
                % 适应度评价
                fitness(i) = Fitness(p(i, :)');
                FEs = FEs + 1;
                
                if fitness(i) < bestever
                    gbestx = p(i, :);
                    bestever = fitness(i);
                end
                
                if FEs >= MaxFEs
                    break;
                end
            end
        end
        % 记录当代最佳
        gbesthistory(iter) = bestever;
    end
    
    % 补齐历史数据
    if iter < maxiter
        gbesthistory(iter+1:maxiter) = bestever;
    elseif iter > maxiter
        gbesthistory(maxiter+1:end) = [];
    end
    
    % 返回最优路径三维格式
    bestPath = decodePath(gbestx);
end
