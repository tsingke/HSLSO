function [bestPath, gbesthistory] = runSCLDPSO(Fitness, popsize, dimension, xmax, xmin, maxiter)
    % 参数初始化
    N = 10;
    % 确保 popsize 能被 N 整除，避免索引越界
    if mod(popsize, N) ~= 0
        m = floor(popsize / N);
        popsize = m * N;
    else
        m = popsize / N;
    end

    % 动态计算速度边界
    vmax = (xmax - xmin) * 0.2;
    vmin = -vmax;
    
    phi = 0.3;
    FEs = 0;
    MaxFEs = popsize * maxiter;
    
    % 初始化种群与速度
    p = xmin + (xmax - xmin) .* rand(popsize, dimension);
    v = vmin + (vmax - vmin) .* rand(popsize, dimension);
    fitness = zeros(popsize, 1);
    
    % 初始评估
    for i = 1:popsize
        fitness(i) = Fitness(p(i, :)');
    end
    FEs = FEs + popsize;
    
    pbest = p;
    pbestfitness = fitness;
    [bestever, id] = min(fitness);
    gbestx = p(id, :);
    
    % 统一历史记录维度为 maxiter
    gbesthistory = zeros(1, maxiter);
    iter = 1;
    gbesthistory(iter) = bestever;
    
    % 主循环
    while FEs < MaxFEs
        iter = iter + 1;
        if iter > maxiter; break; end % 保护机制
        
        % 排序和分层
        [fitness, rank] = sort(fitness);
        p = p(rank, :);
        v = v(rank, :);
        pbest = pbest(rank, :);
        pbestfitness = pbestfitness(rank);
        
        for sub = 1:N
            if sub >= 2
                for k = 1:m
                    i = (sub - 1) * m + k;
                    
                    % 学习范例选择 (采用向量化取法提速)
                    if sub == 2
                        learna = pbest(randperm(m, 1), :);
                        learnb = pbest(randperm(m, 1), :);
                    else
                        r = randperm(sub - 1, 2);
                        a = min(r);
                        b = max(r);
                        learna = pbest((a - 1) * m + randperm(m, 1), :);
                        learnb = pbest((b - 1) * m + randperm(m, 1), :);
                    end
                    
                    % 位置与速度更新
                    v(i, :) = rand(1, dimension) .* v(i, :) + rand(1, dimension) .* (learna - p(i, :)) + phi .* rand(1, dimension) .* (learnb - p(i, :));
                    p(i, :) = p(i, :) + v(i, :);
                    
                    % 边界约束
                    p(i, :) = max(min(p(i, :), xmax), xmin);
                    
                    % 适应度评价
                    fitness(i) = Fitness(p(i, :)');
                    FEs = FEs + 1;
                    
                    % 更新个体最优和全局最优
                    if fitness(i) < pbestfitness(i)
                        pbestfitness(i) = fitness(i);
                        pbest(i, :) = p(i, :);
                    end
                    if fitness(i) < bestever
                        bestever = fitness(i);
                        gbestx = p(i, :);
                    end
                    
                    if FEs >= MaxFEs
                        break;
                    end
                end
            end
            if FEs >= MaxFEs
                break;
            end
        end
        % 记录每一代的全局最优
        gbesthistory(iter) = bestever;
    end
    
    % 补齐未满的迭代历史（修复了原代码 gbestfitness 未定义的报错）
    if iter < maxiter
        gbesthistory(iter+1:maxiter) = bestever;
    elseif iter > maxiter
        gbesthistory(maxiter+1:end) = [];
    end
    
    % 输出格式转换为三维航点矩阵
    bestPath = decodePath(gbestx);
end
