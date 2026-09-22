function [gbestx, bestever, gbesthistory] = zAHLSO(mainHandle, popsize, dimension, xmax, xmin, vmax, vmin, maxiter, fCalculation, FuncId, VisualSwitch)
% AHLSO - Agent-assisted Heterogeneous Learning Swarm Optimizer
% 基于论文: "An agent-assisted heterogeneous learning swarm optimizer 
%            for large-scale optimization" (2024)
% 
% 主要创新:
% 1. 异构学习结构(HL): 局部搜索+全局搜索
% 2. 智能体辅助进化搜索: Q-learning动态选择学习结构
% 3. 层级资源分配(LRA): 动态分配计算资源
% 4. 层级样例选择(LSS): 防止早熟收敛

%% ================ 参数设置 ================
popsize = 500;          % 种群大小 (论文推荐)
phi = 0.35;             % 全局搜索权重 (论文Table 2)
alpha = 0.4;            % Q-learning学习率
gamma = 0.8;            % Q-learning折扣因子
epsilon = 0.6;          % epsilon-greedy探索率

% 学习结构集合 (论文Algorithm 1, line 3)
S_local = [4, 6, 8, 10, 20, 50];    % 局部搜索层级集合
S_global = [4, 6, 8, 10];           % 全局搜索层级集合

% 状态空间: 4个离散状态 (论文3.4.1节)
n_states = 4;

% 初始化Q表 (论文Algorithm 4, lines 1-5)
Q_local = zeros(n_states, length(S_local));   % 目标空间状态 × 局部动作
Q_global = zeros(n_states, length(S_global)); % 决策空间状态 × 全局动作

ComputeFitness = fCalculation;
FEs = 0;
MaxFEs = 3e6;

%% ================ 种群初始化 ================
p = zeros(popsize, dimension);
v = zeros(popsize, dimension);
fitness = zeros(popsize, 1);

for i = 1:popsize
    p(i,:) = xmin + (xmax - xmin) .* rand(1, dimension);
    v(i,:) = vmin + (vmax - vmin) .* rand(1, dimension);
    fitness(i) = ComputeFitness(p(i,:)', FuncId);
end
FEs = FEs + popsize;

[bestever, id] = min(fitness);
gbestx = p(id,:);
gbesthistory = bestever * ones(MaxFEs, 1);

fprintf('AHLSO初始化完成, 初始最佳适应度 = %e\n', bestever);

%% ================ 主循环 ================
gen = 1;

while FEs < MaxFEs
    
    %% 步骤1: 计算参考点 (论文Eq. 8-10)
    best_val = min(fitness);
    worst_val = max(fitness);
    
    % 计算权重 (Eq. 8)
    w = zeros(popsize, 1);
    for i = 1:popsize
        w(i) = best_val / (fitness(i) + 1e-10);
    end
    w_sum = sum(w);
    
    % 计算参考点 (Eq. 9)
    p_ref = zeros(1, dimension);
    for d = 1:dimension
        p_ref(d) = sum(w .* p(:, d)) / w_sum;
    end
    
    % 计算每个粒子到参考点的距离 (Eq. 10)
    dist_ref = zeros(popsize, 1);
    for i = 1:popsize
        dist_ref(i) = sqrt(sum((p(i,:) - p_ref).^2));
    end
    
    %% 步骤2: 种群排序 (论文Algorithm 5, lines 5-6)
    % 按适应度排序 (用于局部搜索) - 升序
    [fitness_sorted, idx_fitness] = sort(fitness, 'ascend');
    P_fitness = p(idx_fitness, :);
    
    % 按参考点距离排序 (用于全局搜索) - 升序
    [dist_sorted, idx_dist] = sort(dist_ref, 'ascend');
    P_reference = p(idx_dist, :);
    
    %% 步骤3: 逐个更新粒子 (从最差到最好)
    for i = popsize:-1:1  % 论文Algorithm 5, line 7: i = NP → 1
        
        % 获取当前粒子在原种群中的索引
        current_idx = idx_fitness(i);
        
        %% 步骤3.1: 计算粒子状态 (论文Algorithm 5, line 9; Eq. 12-13)
        
        % 决策空间状态 (Eq. 12) - 用于全局搜索结构选择
        min_dist = min(dist_ref);
        max_dist = max(dist_ref);
        if max_dist > min_dist
            s_D_norm = (dist_ref(current_idx) - min_dist) / (max_dist - min_dist);
        else
            s_D_norm = 0;
        end
        
        % 映射到4个状态: {nearest, nearer, farther, farthest}
        if s_D_norm <= 0.25
            state_D = 1;  % nearest
        elseif s_D_norm <= 0.5
            state_D = 2;  % nearer
        elseif s_D_norm <= 0.75
            state_D = 3;  % farther
        else
            state_D = 4;  % farthest
        end
        
        % 目标空间状态 (Eq. 13) - 用于局部搜索结构选择
        if worst_val > best_val
            s_O_norm = (fitness(current_idx) - best_val) / (worst_val - best_val);
        else
            s_O_norm = 0;
        end
        
        % 映射到4个状态: {smallest, small, larger, largest}
        if s_O_norm <= 0.25
            state_O = 1;  % smallest
        elseif s_O_norm <= 0.5
            state_O = 2;  % small
        elseif s_O_norm <= 0.75
            state_O = 3;  % larger
        else
            state_O = 4;  % largest
        end
        
        %% 步骤3.2: Agent选择动作 (论文Algorithm 5, line 10; Eq. 14)
        
        % 选择局部搜索结构 (基于目标空间状态)
        if rand < epsilon
            action_L = randi(length(S_local));  % 探索
        else
            [~, action_L] = max(Q_local(state_O, :));  % 利用
        end
        SL = S_local(action_L);
        
        % 选择全局搜索结构 (基于决策空间状态)
        if rand < epsilon
            action_G = randi(length(S_global));  % 探索
        else
            [~, action_G] = max(Q_global(state_D, :));  % 利用
        end
        SG = S_global(action_G);
        
        %% 步骤3.3: 计算当前粒子所在层级 (论文Algorithm 5, lines 11-14)
        
        % 在局部搜索结构中的层级 (基于适应度排序)
        level_L = ceil(i / (popsize / SL));
        level_L = min(level_L, SL);  % 确保不超过最大层级
        
        % 在全局搜索结构中的层级 (基于距离排序)
        rank_in_dist = find(idx_dist == current_idx, 1);
        level_G = ceil(rank_in_dist / (popsize / SG));
        level_G = min(level_G, SG);  % 确保不超过最大层级
        
        %% 步骤3.4: 层级资源分配 (LRA) (论文Algorithm 5, lines 15-19; Eq. 17)
        
        % 最优层直接保留,不消耗资源
        if level_L == 1
            continue;
        end
        
        % 计算资源分配概率 (Eq. 17)
        prob_i = (level_L / SL) - (1 - 2*level_L/SL) * (FEs / MaxFEs);
        
        % 概率性分配资源
        if rand > prob_i
            continue;  % 不分配资源,粒子直接进入下一代
        end
        
        %% 步骤3.5: 生成局部学习样例 (论文Algorithm 2)
        
        LG_local = ceil(popsize / SL);  % 每层的粒子数
        
        if level_L >= 2
            % 从前(level_L - 1)层随机选择两层 (Algorithm 2, line 3)
            m = randi(level_L - 1);
            n = randi(level_L - 1);
            m = min(m, n);  % 选择更好的层 (Algorithm 2, line 4)
            
            % 从选定层随机选择一个粒子 (Algorithm 2, line 5)
            start_idx = (m - 1) * LG_local + 1;
            end_idx = min(m * LG_local, popsize);
            k1 = randi([start_idx, end_idx]);
            p_local = P_fitness(k1, :);
        else
            % level_L == 1的情况在前面已经continue了
            % 这里不会执行到
            p_local = P_fitness(1, :);
        end
        
        %% 步骤3.6: 生成全局学习样例 (论文Algorithm 3 + LSS策略)
        
        LG_global = ceil(popsize / SG);  % 每层的粒子数
        
        % 层级样例选择策略 (LSS) - Algorithm 5, line 21
        max_attempts = 20;
        p_global = p_local;  % 默认值
        
        for attempt = 1:max_attempts
            % 从前level_G层随机选择一层 (Algorithm 3, line 4)
            m = randi(level_G);
            
            % 从该层随机选择一个粒子 (Algorithm 3, line 5)
            start_idx = (m - 1) * LG_global + 1;
            end_idx = min(m * LG_global, popsize);
            k1 = randi([start_idx, end_idx]);
            p_global_candidate = P_reference(k1, :);
            
            % 确保全局样例与局部样例和当前粒子不同 (Algorithm 3, line 2)
            if ~isequal(p_global_candidate, p_local) && ...
               ~isequal(p_global_candidate, p(current_idx, :))
                p_global = p_global_candidate;
                break;
            end
        end
        
        %% 步骤3.7: 更新粒子 (论文Eq. 6-7)
        
        r1 = rand(1, dimension);
        r2 = rand(1, dimension);
        r3 = rand(1, dimension);
        
        % 速度更新 (Eq. 6)
        v(current_idx, :) = r1 .* v(current_idx, :) + ...
                            r2 .* (p_local - p(current_idx, :)) + ...
                            phi * r3 .* (p_global - p(current_idx, :));
        
        % 速度边界处理
        v(current_idx, :) = max(v(current_idx, :), vmin);
        v(current_idx, :) = min(v(current_idx, :), vmax);
        
        % 位置更新 (Eq. 7)
        p(current_idx, :) = p(current_idx, :) + v(current_idx, :);
        
        % 位置边界处理
        p(current_idx, :) = max(p(current_idx, :), xmin);
        p(current_idx, :) = min(p(current_idx, :), xmax);
        
        % 计算新适应度 (Algorithm 5, line 22)
        fitness_old = fitness(current_idx);
        fitness(current_idx) = ComputeFitness(p(current_idx, :)', FuncId);
        FEs = FEs + 1;
        
        %% 步骤3.8: 计算奖励 (论文Algorithm 5, line 24; Eq. 15-16)
        
        % 局部搜索奖励 (Eq. 15) - 基于适应度改善
        if fitness_old ~= 0
            r_local = (fitness_old - fitness(current_idx)) / abs(fitness_old);
        else
            r_local = fitness_old - fitness(current_idx);
        end
        
        % 全局搜索奖励 (Eq. 16) - 基于到参考点距离变化
        dist_new = sqrt(sum((p(current_idx, :) - p_ref).^2));
        
        if r_local > 0  % 如果适应度改善了
            % 奖励靠近参考点的移动
            r_global = abs(dist_ref(current_idx) - dist_new) / ...
                       (max_dist - min_dist + 1e-10);
        else
            % 否则根据距离变化给奖励
            r_global = (dist_ref(current_idx) - dist_new) / ...
                       (max_dist - min_dist + 1e-10);
        end
        
        %% 步骤3.9: 更新Q表 (论文Algorithm 5, line 25; Eq. 11)
        
        % 计算新状态
        if max_dist > min_dist
            s_D_norm_new = (dist_new - min_dist) / (max_dist - min_dist);
        else
            s_D_norm_new = 0;
        end
        
        if s_D_norm_new <= 0.25
            state_D_next = 1;
        elseif s_D_norm_new <= 0.5
            state_D_next = 2;
        elseif s_D_norm_new <= 0.75
            state_D_next = 3;
        else
            state_D_next = 4;
        end
        
        if worst_val > best_val
            s_O_norm_new = (fitness(current_idx) - best_val) / (worst_val - best_val);
        else
            s_O_norm_new = 0;
        end
        
        if s_O_norm_new <= 0.25
            state_O_next = 1;
        elseif s_O_norm_new <= 0.5
            state_O_next = 2;
        elseif s_O_norm_new <= 0.75
            state_O_next = 3;
        else
            state_O_next = 4;
        end
        
        % Q-learning更新 (Bellman方程, Eq. 11)
        % Q(s_t, a_t) = Q(s_t, a_t) + α·(r + γ·Max(Q(s_{t+1})) - Q(s_t, a_t))
        
        Q_local(state_O, action_L) = Q_local(state_O, action_L) + ...
            alpha * (r_local + gamma * max(Q_local(state_O_next, :)) - ...
            Q_local(state_O, action_L));
        
        Q_global(state_D, action_G) = Q_global(state_D, action_G) + ...
            alpha * (r_global + gamma * max(Q_global(state_D_next, :)) - ...
            Q_global(state_D, action_G));
        
        %% 步骤3.10: 更新全局最优
        if fitness(current_idx) < bestever
            bestever = fitness(current_idx);
            gbestx = p(current_idx, :);
        end
        
        gbesthistory(FEs) = bestever;
        
        % 输出进度
        if mod(FEs, 10000) == 0
            fprintf('Gen %d, FEs=%d, Best=%.6e, AvgFit=%.6e\n', ...
                    gen, FEs, bestever, mean(fitness));
        end
        
        % 检查终止条件
        if FEs >= MaxFEs
            break;
        end
    end
    
    gen = gen + 1;
    
    % 检查终止条件
    if FEs >= MaxFEs
        break;
    end
end

%% ================ 填充历史记录 ================
if FEs < MaxFEs
    gbesthistory(FEs+1:MaxFEs) = bestever;
elseif FEs > MaxFEs
    gbesthistory = gbesthistory(1:MaxFEs);
end

fprintf('\n========================================\n');
fprintf('AHLSO算法完成!\n');
fprintf('总评估次数: %d\n', FEs);
fprintf('最终最佳适应度: %.6e\n', bestever);
fprintf('========================================\n');

end