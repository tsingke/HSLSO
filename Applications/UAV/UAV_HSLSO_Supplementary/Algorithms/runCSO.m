function [bestPath, gbesthistory] = runCSO(Fitness, popsize, dimension, xmax, xmin, maxiter)
    % 1. 参数初始化
    m = popsize; % 统一使用传入的种群规模
    phi = 0.1;
    
    % 速度边界自适应
    vmax = (xmax - xmin) * 0.2;
    vmin = -vmax;
    
    FEs = 0;
    MaxFEs = popsize * maxiter;
    
    % 2. 初始化种群和速度
    p = xmin + (xmax - xmin) .* rand(m, dimension);
    v = vmin + (vmax - vmin) .* rand(m, dimension);

    fitness = zeros(m, 1);
    
    % 初始评估
    for i = 1:m
        fitness(i) = Fitness(p(i,:)');
    end
    FEs = FEs + m;
    
    [bestever, id] = min(fitness);
    gbestx = p(id, :);
    
    % 初始化历史记录（严格映射到 maxiter 长度以对齐画图）
    gbesthistory = zeros(1, maxiter);
    gbesthistory(1) = bestever;
    
    % 3. 主循环
    while FEs < MaxFEs
        % 随机构建竞争对
        rlist = randperm(m);
        % 使用 floor 处理可能的奇数 popsize 情况
        half_m = floor(m / 2);
        rpairs = [rlist(1:half_m); rlist(half_m + 1 : 2 * half_m)]';
        
        % 计算当前种群中心位置
        center = mean(p);
        
        % 粒子对竞争
        mask = (fitness(rpairs(:, 1)) > fitness(rpairs(:, 2)));
        for k = 1:half_m
            if mask(k) == 0
                los = rpairs(k, 2);
                win = rpairs(k, 1);
            else
                los = rpairs(k, 1);
                win = rpairs(k, 2);
            end
            
            % 失败者向胜利者和中心位置学习
            v(los, :) = rand(1, dimension) .* v(los, :) + ...
                        rand(1, dimension) .* (p(win, :) - p(los, :)) + ...
                        phi * rand(1, dimension) .* (center - p(los, :));
            p(los, :) = p(los, :) + v(los, :);
            
            % 边界约束
            p(los, :) = max(min(p(los, :), xmax), xmin);
            
            % 适应度评价（只有失败者需要重新评价）
            fitness(los) = Fitness(p(los, :)');
            FEs = FEs + 1;
            
            % 更新全局最优
            if fitness(los) < bestever
                bestever = fitness(los);
                gbestx = p(los, :);
            end
            
            % 精确映射当前 FEs 对应的标准代数索引，保障绘图对齐
            idx = min(maxiter, floor(FEs / popsize) + 1);
            gbesthistory(idx) = bestever;
            
            if FEs >= MaxFEs
                break;
            end
        end
    end
    
    % 补齐历史数据（防止因提前跳出导致末尾出现 0）
    for i = 2:maxiter
        if gbesthistory(i) == 0
            gbesthistory(i) = gbesthistory(i-1);
        end
    end
    
    % 返回最优路径三维格式
   bestPath = decodePath(gbestx);
end
