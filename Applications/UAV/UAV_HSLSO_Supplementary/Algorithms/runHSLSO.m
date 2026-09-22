function [bestPath, gbesthistory] = runHSLSO(Fitness, popsize, dimension, xmax, xmin, maxiter)
    % 1. 参数初始化
    N = 10; 
    % 确保 popsize 能被 N 整除，避免索引越界
    if mod(popsize, N) ~= 0
        m = floor(popsize / N);
        popsize = m * N;
    else
        m = popsize / N;
    end
    
    % 速度边界自适应
    vmax = (xmax - xmin) * 0.2; 
    vmin = -vmax;
    
    phi = 0.3;
    FEs = 0;
    MaxFEs = popsize * maxiter;
    PLinit = (1:N)/N;
    PLfina = 1 - PLinit;
    
    % 初始化种群与速度
    p = xmin + (xmax - xmin) .* rand(popsize, dimension);
    v = vmin + (vmax - vmin) .* rand(popsize, dimension);
    fitness = zeros(popsize, 1);
    
    % 初始评估
    for i = 1:popsize
        fitness(i) = Fitness(p(i,:)');    
    end
    FEs = FEs + popsize;
    
    pbest = p;
    pbestfitness = fitness;
    [bestever, id] = min(fitness);
    gbestx = p(id, :); 
    
    % 初始化历史记录（按代数记录，对齐主框架作图）
    gbesthistory = zeros(1, maxiter);
    iter = 1;
    gbesthistory(iter) = bestever;
    
    % 2. 主循环
    while FEs < MaxFEs
        iter = iter + 1;
        if iter > maxiter; break; end % 保护机制
        
        % 排序和分层
        [fitness, rank] = sort(fitness);
        p = p(rank,:);
        v = v(rank,:);
        pbest = pbest(rank,:);
        pbestfitness = pbestfitness(rank);
        
        for sub = 1:N
            if sub >= 2
                PL(sub) = PLinit(sub) + (PLfina(sub) - PLinit(sub)) * (FEs / MaxFEs);
                
                for k = 1:m
                    if rand < PL(sub)
                        i = (sub - 1) * m + k; 
                        
                        % 选择学习范例
                        if sub == 2
                            learna = pbest(randperm(m, 1), :);
                            learnb = pbest(randperm(m, 1), :);
                        else
                            k_layers = ceil((sub - 1) * (1 - (FEs / MaxFEs)^2));
                            if k_layers < 2
                                k_layers = 2;
                            end
                            
                            r = randperm(k_layers, 2);
                            a = min(r);
                            b = max(r);
                       
                            learna = pbest((a - 1) * m + randperm(m, 1), :);
                            learnb = pbest((b - 1) * m + randperm(m, 1), :);
                        end
                        
                        % 位置更新
                        v(i,:) = rand(1,dimension).*v(i,:) + rand(1,dimension).*(learna-p(i,:)) + phi.*rand(1,dimension).*(learnb-p(i,:));
                        p(i,:) = p(i,:) + v(i,:);
                        
                        % 边界越界处理
                        p(i,:) = max(min(p(i,:), xmax), xmin);
                        
                        % 适应度评价
                        fitness(i) = Fitness(p(i,:)');
                        FEs = FEs + 1;
                        
                        % pbest 及 gbest 更新
                        if fitness(i) < pbestfitness(i)
                            pbestfitness(i) = fitness(i);
                            pbest(i,:) = p(i,:);
                        end
                        
                        if fitness(i) < bestever
                            bestever = fitness(i);
                            gbestx = p(i,:);
                        end
                        
                        if FEs >= MaxFEs
                            break;
                        end
                    end
                end
            end
            if FEs >= MaxFEs
                break;
            end
        end
        % 记录当代最佳
        gbesthistory(iter) = bestever;
    end
    
    % 补齐历史（如果提早触发 FEs 结束）
    if iter < maxiter
        gbesthistory(iter+1:maxiter) = bestever; 
    elseif iter > maxiter
        gbesthistory(maxiter+1:end) = []; 
    end
    
    % 输出格式转换为三维航点矩阵
    bestPath = decodePath(gbestx);
end
