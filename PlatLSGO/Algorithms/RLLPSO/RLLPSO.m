function [gbestX, gbestfitness, gbesthistory] = RLLPSO(mainHandle, PopSize, D, xmax, xmin, vmax, vmin, MaxIter, fCal, FuncId, VisualSwitch)
% RLLPSO: Reinforcement Learning Level-based PSO for LSOPs
% 平台接口: [gbestX, gbestfitness, gbesthistory] = RLLPSO(mainHandle, PopSize, D, xmax, xmin, vmax, vmin, MaxIter, fCal, FuncId, VisualSwitch)
% 论文设置:  φ=0.4, α=0.4, γ=0.8, ε=0.9, S={4,6,8,10,20,50}; p_lcomp = (FEcount/MaxFEs)^2
% FEs 计数严格对齐平台: MaxFEs = MaxIter * PopSize; 每评价一个个体 +1，并写入 gbesthistory(FEs)
PopSize=500;
% -------------------- 参数/常量 --------------------
phi = 0.4;           % φ in Eq.(4)
alpha = 0.4;         % Q-learning α
gamma = 0.8;         % Q-learning γ
epsilon = 0.9;       % ε-greedy
S_cand = [4 6 8 10 20 50];              % 层数候选集合
S_valid = S_cand(S_cand <= PopSize);    % 不能超过种群规模
if isempty(S_valid), S_valid = min(4, PopSize); end
if numel(S_valid)==1, epsilon = 1; end  % 只有一个动作时总是“利用”

MaxFEs = 3*10^6;
FEs = 0;
tiny = 1e-12;        % 防除零

% -------------------- 初始化群体 --------------------
% 位置/速度
X = xmin + (xmax - xmin) .* rand(PopSize, D);
V = zeros(PopSize, D);

% 初次评估
fitness = inf(PopSize,1);
gbestfitness = inf; 
gbestX = zeros(1, D);
for i = 1:PopSize
    fi = fCal(X(i,:)', FuncId);
    fitness(i) = fi;
    if fi < gbestfitness
        gbestfitness = fi;
        gbestX = X(i,:);
    end
end

% 历史记录
gbesthistory = inf(1, MaxFEs);

% Q 表（状态、动作均为“层数索引”）
ns = numel(S_valid);
Q = zeros(ns, ns);
state_idx = 1;                    % s0 = l1
prev_gbest = gbestfitness;        % 计算奖励时用

% -------------------- 主循环（按FEs推进） --------------------
while FEs <= MaxFEs
    % 1) ε-greedy 选择层数动作（对应下一轮的 level 数）
    if rand < epsilon
        [~, action_idx] = max(Q(state_idx, :));   % 利用
    else
        action_idx = randi(ns);                   % 探索
    end
    L = S_valid(action_idx);       % 本轮层数
    
    % 2) 按适应度排序并等分为 L 个层（最后一层承接余数，且为最差层）
    [~, ord] = sort(fitness, 'ascend');           % 小为优
    levelIdx = partition_levels(ord, PopSize, L); % cell{1..L}, 每个是索引数组
    
    % 3) 生成新一代（基于层间学习与层间竞争）
    Xnew = X;
    Vnew = V;
    % 3.1 第2层: 仅向第1层学习；第1层: 冻结
    if L >= 2 && ~isempty(levelIdx{2})
        ex1_pool = levelIdx{1};
        for k = levelIdx{2}(:)'   % 逐个粒子
            [e1_idx, e2_idx] = pick_two_from_level(ex1_pool);
            r1 = rand(1, D); r2 = rand(1, D); r3 = rand(1, D);
            Vnew(k,:) = r1 .* V(k,:) + r2 .* (X(e1_idx,:) - X(k,:)) + phi .* r3 .* (X(e2_idx,:) - X(k,:));
            Vnew(k,:) = clip_by(Vnew(k,:), vmin, vmax);
            Xnew(k,:) = X(k,:) + Vnew(k,:);
            Xnew(k,:) = min(max(Xnew(k,:), xmin), xmax);
        end
    end
    % 3.2 第3..L层: 向两个“更高层”学习，带层间竞争
    for li = 3:L
        cur_pool = levelIdx{li};
        if isempty(cur_pool), continue; end
        for k = cur_pool(:)'   % 逐个粒子
            % 选择两个示例层（Algorithm 2: Level competition mechanism）
            probl = (FEs / MaxFEs)^2; % 用当前已用FEs估计触发概率
            [le1, le2] = select_two_exemplar_levels(li, L, probl);
            % 从层 le1, le2 各随机取 1 个示例粒子
            e1 = levelIdx{le1}( randi(numel(levelIdx{le1})) );
            e2 = levelIdx{le2}( randi(numel(levelIdx{le2})) );
            % 更新速度/位置 (Eq.4 & Eq.5)
            r1 = rand(1, D); r2 = rand(1, D); r3 = rand(1, D);
            Vnew(k,:) = r1 .* V(k,:) + r2 .* (X(e1,:) - X(k,:)) + phi .* r3 .* (X(e2,:) - X(k,:));
            Vnew(k,:) = clip_by(Vnew(k,:), vmin, vmax);
            Xnew(k,:) = X(k,:) + Vnew(k,:);
            Xnew(k,:) = min(max(Xnew(k,:), xmin), xmax);
        end
    end
    
    % 4) 评估新种群并记录（与平台 GA 规范一致：每次评价 +1，并写 gbesthistory）
    newfitness = zeros(PopSize,1);
    for i = 1:PopSize
        if FEs >= MaxFEs, break; end
        FEs = FEs + 1;
        fi = fCal(Xnew(i,:)', FuncId);
        newfitness(i) = fi;
        if fi < gbestfitness
            gbestfitness = fi;
            gbestX = Xnew(i,:);
        end
        gbesthistory(FEs) = gbestfitness;
        if mod(FEs, floor(MaxFEs/10)) == 0 && FEs <= MaxFEs
            fprintf('RLLPSO 第%d次评价，最佳适应度 = %e\n', FEs, gbestfitness);
        end
    end
    
    % 5) 环境反馈（奖励）与 Q 更新（Eq.7 & Eq.8）
    reward = abs(prev_gbest - gbestfitness) / max(abs(prev_gbest), tiny);
    prev_gbest = gbestfitness;
    if ns > 1
        Q(state_idx, action_idx) = Q(state_idx, action_idx) + ...
            alpha * (reward + gamma * max(Q(action_idx, :)) - Q(state_idx, action_idx));
        state_idx = action_idx; % s_{t+1} = a_{t+1}
    end
    
    % 6) 准备下一轮
    X = Xnew; V = Vnew; fitness = newfitness;
    
    % 7) 循环结束条件（按FEs）
    if FEs >= MaxFEs, break; end
end

% 末尾补齐（平台要求 gbesthistory 长度恰为 MaxFEs）
if FEs < MaxFEs
    gbesthistory(FEs+1:MaxFEs) = gbestfitness;
elseif FEs > MaxFEs
    gbesthistory(MaxFEs+1:end) = [];
end

end % ===== end of main function =====

% -------------------- 辅助函数 --------------------
function cells = partition_levels(order_idx, N, L)
% 将排序后的索引均分为 L 层；最后一层承接余数（且为“最差层”）
LS = floor(N / L);
remN = mod(N, L);
cells = cell(L,1);
startPos = 1;
for li = 1:L
    take = LS + (li==L) * remN;
    if take == 0
        cells{li} = [];
    else
        cells{li} = order_idx(startPos : startPos + take - 1);
        startPos = startPos + take;
    end
end
end

function [le1, le2] = select_two_exemplar_levels(curL, L, probl)
% Algorithm 2：层间竞争。返回两个“更高层”（更精英：数字更小）
le1 = choose_one_level(curL, L, probl);
le2 = choose_one_level(curL, L, probl);
% 若 le2 比 le1 更高（更精英），交换，保证 le1 不比 le2 精英
if le2 < le1
    tmp = le1; le1 = le2; le2 = tmp;
end
end

function le = choose_one_level(curL, L, probl)
% 从 {1,2,...,curL-1} 中选择一层；以概率 probl 触发“两层竞赛，取更高层”
if curL <= 2
    le = 1; return;
end
if rand < probl && (curL - 1) >= 2
    c1 = randi(curL-1); c2 = randi(curL-1);
    while c2 == c1, c2 = randi(curL-1); end
    le = min(c1, c2); % 更“高”的层（数字更小）胜出
else
    le = randi(curL-1);
end
end

function [i1, i2] = pick_two_from_level(pool)
% 从同一层随机取两个示例粒子（可相同）
n = numel(pool);
if n == 1
    i1 = pool(1); i2 = pool(1);
else
    i1 = pool(randi(n));
    i2 = pool(randi(n));
end
end

function v = clip_by(v, vmin, vmax)
% 速度裁剪（若平台未实际使用 vmax/vmin，可保持零或无穷）
if ~isempty(vmax) && ~isempty(vmin)
    v = min(max(v, vmin), vmax);
end
end
