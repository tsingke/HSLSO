function [gbestx, bestever, gbesthistory] = RCIPSO(mainHandle, popsize, dimension, xmax, xmin, vmax, vmin, maxiter, fCalculation, FuncId, VisualSwitch)

% --- 参数初始化 ---
popsize = 800;
% dimension = 1000;
phi = 0.3;

% --- 框架变量 ---
ComputeFitness = fCalculation;
FEs = 0;
MaxFEs = 3000000;

% --- 种群初始化 ---
p = zeros(popsize, dimension);
v = zeros(popsize, dimension);
fitness = zeros(popsize, 1);
for i = 1:popsize
    p(i,:) = xmin + (xmax - xmin) .* rand(1, dimension);
    v(i,:) = zeros(1, dimension);
    fitness(i) = ComputeFitness(p(i,:)', FuncId);
end
FEs = FEs + popsize;
[bestever, id] = min(fitness);
gbestx = p(id,:); 
gbesthistory = zeros(MaxFEs, 1); 
gbesthistory(1:FEs) = bestever;

% --- 主循环 ---
while FEs < MaxFEs
    topology_size = floor(2 + (10 - 2) * sqrt(FEs / MaxFEs));

    for i = 1:popsize
        if FEs >= MaxFEs, break; end

        population_indices = 1:popsize;
        population_indices(i) = [];
        shuffled_indices = population_indices(randperm(popsize - 1));
        topo_indices = shuffled_indices(1:topology_size);

        topo_fitness = fitness(topo_indices);
        [sorted_topo_fitness, sort_idx] = sort(topo_fitness);
        sorted_topo_indices = topo_indices(sort_idx);

        if sorted_topo_fitness(2) <= fitness(i)
            exemplar1_idx = sorted_topo_indices(1);

            % --- exemplar2 选择逻辑 (已修复bug) ---
            exemplar2_idx = -1;
            for j = topology_size:-1:1
                if sorted_topo_fitness(j) <= fitness(i)
                    exemplar2_idx = sorted_topo_indices(j);
                    break;
                end
            end

            % 【关键修正】确保exemplar2_idx总是一个有效索引
            if exemplar2_idx == -1
                exemplar2_idx = sorted_topo_indices(topology_size);
            end

            % --- 速度和位置更新 (保留性能更好的向量化随机数) ---
            r1 = rand(1, dimension);
            r2 = rand(1, dimension);
            r3 = rand(1, dimension);

            v(i,:) = r1 .* v(i,:) + ...
                     r2 .* (p(exemplar1_idx, :) - p(i, :)) + ...
                     phi .* r3 .* (p(exemplar2_idx, :) - p(i, :));

            p(i,:) = p(i,:) + v(i,:);

            p(i,:) = max(p(i,:), xmin);
            p(i,:) = min(p(i,:), xmax);

            fitness(i) = ComputeFitness(p(i,:)', FuncId);
            FEs = FEs + 1;

            if fitness(i) < bestever
                bestever = fitness(i);
                gbestx = p(i,:);
            end

            if FEs <= MaxFEs
                gbesthistory(FEs) = bestever;
                 if mod(FEs, floor(MaxFEs/10)) == 0 && FEs <= MaxFEs
                 fprintf("RCIPSO算法,第%d次评价，最佳适应度 = %e，拓扑大小 = %d\n", FEs, bestever, topology_size);
                 end
            end
        end
    end
end

% --- 后处理 ---
if FEs < MaxFEs
    gbesthistory(FEs+1:end) = bestever;
else
    if length(gbesthistory) > MaxFEs
        gbesthistory = gbesthistory(1:MaxFEs);
    end
end
end




% function [gbestx, bestever, gbesthistory] = rcipso_corrected(mainHandle, popsize, dimension, xmax, xmin, vmax, vmin, maxiter, fCalculation, FuncId, VisualSwitch)
% % RCIPSO - 严格按照C++代码逻辑修正版本
% 
% % --- 参数初始化（完全对应C++） ---
% popsize = 800;              % const int populationsize = 800
% dimension = 1000;           % const int dim = 1000
% phi = 0.3;                  % const double phi = 0.3
% 
% % --- 框架变量 ---
% ComputeFitness = fCalculation;
% FEs = 0;
% MaxFEs = 3000 * dimension;  % const int MaxFEs= 3000*dim
% 
% % --- 种群初始化（对应initialization函数） ---
% p = zeros(popsize, dimension);
% v = zeros(popsize, dimension);
% fitness = zeros(popsize, 1);
% 
% for i = 1:popsize
%     p(i,:) = xmin + (xmax - xmin) .* rand(1, dimension);
%     v(i,:) = zeros(1, dimension);  % 对应C++中的v[i][j]= 0
% end
% 
% % --- 初始适应度计算（对应Fitness_Computation函数） ---
% fitness(1) = ComputeFitness(p(1,:)', FuncId);
% bestever = fitness(1);
% gl_best = 1;
% FEs = FEs + 1;
% 
% for i = 2:popsize
%     fitness(i) = ComputeFitness(p(i,:)', FuncId);
%     if fitness(i) < bestever
%         bestever = fitness(i);
%         gl_best = i;
%     end
%     FEs = FEs + 1;
% end
% 
% gbestx = p(gl_best,:);
% final_val = bestever;
% 
% % --- 历史记录初始化 ---
% gbesthistory = zeros(MaxFEs, 1);
% gbesthistory(1:FEs) = bestever;
% 
% % --- 主循环 ---
% while FEs <= MaxFEs
%     % 拓扑大小计算（完全对应C++公式）
%     topology_size = floor(2 + (10 - 2) * power((1.0*FEs)/MaxFEs, 0.5));
% 
%     for i = 1:popsize
%         if FEs > MaxFEs, break; end
% 
%         % --- 关键修正：邻域选择逻辑（严格对应C++） ---
%         % 创建并随机打乱种群索引
%         population_index = randperm(popsize);
% 
%         % 获取邻域（对应C++中的复杂逻辑）
%         topology_ids = zeros(topology_size, 1);
%         topology_fitness = zeros(topology_size, 1);
% 
%         m = 1;
%         for j = 1:topology_size
%             % 跳过当前粒子（对应C++逻辑）
%             if population_index(m) ~= i
%                 topology_ids(j) = population_index(m);
%                 topology_fitness(j) = fitness(population_index(m));
%                 m = m + 1;
%             else
%                 m = m + 1;
%                 if m <= popsize
%                     topology_ids(j) = population_index(m);
%                     topology_fitness(j) = fitness(population_index(m));
%                     m = m + 1;
%                 else
%                     % 边界处理
%                     topology_ids(j) = population_index(1);
%                     topology_fitness(j) = fitness(population_index(1));
%                     m = 2;
%                 end
%             end
%         end
% 
%         % 排序拓扑
%         [sorted_fitness, sort_idx] = sort(topology_fitness);
%         sorted_ids = topology_ids(sort_idx);
% 
%         % 更新条件检查（对应C++中的if(topology[1].data <= results[i])）
%         if length(sorted_fitness) >= 2 && sorted_fitness(2) <= fitness(i)
%             % exemplar1选择（最优邻居）
%             exemplar1_idx = sorted_ids(1);
% 
%             % --- 关键修正：exemplar2选择逻辑 ---
%             % 对应C++：从后往前找第一个比当前粒子好的邻居
%             j = topology_size;
%             while j >= 1 && sorted_fitness(j) > fitness(i)
%                 j = j - 1;
%             end
%             exemplar2_idx = sorted_ids(j);
% 
%             % --- 速度和位置更新 ---
%             % 逐维度生成随机数（更严格对应C++）
%             for d = 1:dimension
%                 r1 = rand();
%                 r2 = rand();
%                 r3 = rand();
% 
%                 v(i,d) = r1 * v(i,d) + ...
%                          r2 * (p(exemplar1_idx, d) - p(i, d)) + ...
%                          phi * r3 * (p(exemplar2_idx, d) - p(i, d));
% 
%                 p(i,d) = p(i,d) + v(i,d);
% 
%                 % 边界处理（对应C++严格检查）
%                 if p(i,d) < xmin
%                     p(i,d) = xmin;
%                 end
%                 if p(i,d) > xmax
%                     p(i,d) = xmax;
%                 end
%             end
% 
%             % 函数评价
%             FEs = FEs + 1;
%             fitness(i) = ComputeFitness(p(i,:)', FuncId);
% 
%             % 更新全局最优
%             if fitness(i) < final_val
%                 final_val = fitness(i);
%                 bestever = final_val;
%                 gbestx = p(i,:);
%             end
% 
%             % 记录历史
%             if FEs <= MaxFEs
%                 gbesthistory(FEs) = bestever;
%                 fprintf("RCIPSO算法,第%d次评价，最佳适应度 = %e，拓扑大小 = %d\n", FEs, bestever, topology_size);
%             end
%         end
%     end
% end
% 
% % --- 后处理 ---
% if FEs < MaxFEs
%     gbesthistory(FEs+1:MaxFEs) = bestever;
% else
%     if length(gbesthistory) > MaxFEs
%         gbesthistory = gbesthistory(1:MaxFEs);
%     end
% end
% 
% end





% function [gbestx, bestever, gbesthistory] = RCIPSO(mainHandle, popsize, dimension, xmax, xmin, vmax, vmin, maxiter, fCalculation, FuncId, VisualSwitch)
% 
% 
% % ---------------- 参数与基础变量 ----------------
% popsize=800;
% phi     = 0.3;                 % 对应 const double phi = 0.3
% MaxFEs  = 3000 * dimension;    % 对应 const int MaxFEs = 3000*dim
% FEs     = 0;
% 
% % 记录点 (与 C++ 中 record[i] = 5e5*(i+1) 对应)
% recordPoints = (5e5:5e5:5e5*10);
% recordCount  = numel(recordPoints);
% recordIndex  = 1;
% checkpointBest = zeros(recordCount,1);  %#ok<NASGU> % 若需要可对外输出
% 
% % ---------------- 初始化（对应 initialization + Fitness_Computation） ----------------
% % 位置：均匀随机；速度：0
% position = xmin + (xmax - xmin) .* rand(popsize, dimension);
% velocity = zeros(popsize, dimension);
% 
% fitness  = zeros(popsize,1);
% 
% % 初始全种群评估 (对应 Fitness_Computation)
% % C++ 中: FEs 初始增加 populationsize 次(每个粒子一次)
% for i = 1:popsize
%     fitness(i) = fCalculation(position(i,:)', FuncId);
% end
% FEs = FEs + popsize;
% 
% % 找初始全局最优
% [bestever, gl_best] = min(fitness);
% gbestx = position(gl_best,:);
% 
% % gbesthistory：长度固定为 MaxFEs
% gbesthistory = zeros(MaxFEs,1);
% % 填充已经使用的 FEs（这里初始 FEs=popsize，将前 popsize 次评价的 best 填入）
% gbesthistory(1:FEs) = bestever;
% 
% % population_index 数组 (C++ 中每轮都重排，这里直接临时 randperm)
% % 不需要单独持久化
% % topology 结构在 MATLAB 中用两个数组表示: topo_ids, topo_fits
% 
% if VisualSwitch
%     fprintf('RCIPSO (Func %d) init: FEs=%d best=%e\n', FuncId, FEs, bestever);
% end
% 
% % ---------------- 主循环 (对应 while(FEs <= MaxFEs)) ----------------
% while FEs <= MaxFEs
%     rng('shuffle'); 
%     % 动态拓扑规模 (C++: int, 会向下截断)
%     rawTopo = 2 + (10 - 2) * ((FEs / MaxFEs)^0.5);
%     topology_size = floor(rawTopo);
%     if topology_size < 2
%         topology_size = 2;
%     elseif topology_size > 10
%         topology_size = 10;
%     end
% 
%     % 遍历每个粒子
%     for i = 1:popsize
%         if FEs > MaxFEs
%             break;
%         end
% 
%         % ---------------- 构造拓扑 (严格模拟 C++ “跳过自身再取下一个” 逻辑) ----------------
%         perm = randperm(popsize);   % random_shuffle
%         topo_ids  = zeros(topology_size,1);
%         topo_fits = zeros(topology_size,1);
% 
%         m = 1;
%         for j = 1:topology_size
%             if perm(m) ~= i
%                 topo_ids(j)  = perm(m);
%                 topo_fits(j) = fitness(perm(m));
%                 m = m + 1;
%             else
%                 m = m + 1;  % 跳过自己
%                 % C++ 代码此处潜在越界风险: 若 m 超过 popsize 直接访问会越界
%                 % 安全处理：若越界则重新环回
%                 if m > popsize
%                     m = 1;
%                     % 如果环回又遇到自己 再往后移动
%                     if perm(m) == i
%                         m = m + 1;
%                         if m > popsize
%                             m = 1; % 极端情况下处理
%                         end
%                     end
%                 end
%                 topo_ids(j)  = perm(m);
%                 topo_fits(j) = fitness(perm(m));
%                 m = m + 1;
%                 if m > popsize
%                     m = 1;
%                 end
%             end
%         end
% 
%         % ---------------- 排序（对应 sort(topology, topology+topology_size, Compare_NewType)） ----------------
%         [sorted_fits, idx_in_sorted] = sort(topo_fits);
%         sorted_ids = topo_ids(idx_in_sorted);
% 
%         % ---------------- 更新条件: 第二优邻居的适应度 <= 当前粒子 ----------------
%         if sorted_fits(2) <= fitness(i)
% 
%             exemplar1 = sorted_ids(1);  % 最优邻居
% 
%             % exemplar2: 从末尾往前找第一个 <= 当前粒子适应度 的邻居
%             jj = topology_size;
%             while jj >= 1 && fitness(sorted_ids(jj)) > fitness(i)
%                 jj = jj - 1;
%             end
%             % 进入此 if 块时理论保证至少有两个邻居 <= 当前，因此 jj >=1 一定成立
%             exemplar2 = sorted_ids(jj);
% 
%             % ---------------- 速度 & 位置更新（逐维，使用独立随机数，匹配 C++ 逐次 random_real_num_r() 调用） ----------------
%             for d = 1:dimension
%                 r1 = rand();
%                 r2 = rand();
%                 r3 = rand();
%                 velocity(i,d) = r1 * velocity(i,d) ...
%                               + r2 * (position(exemplar1,d) - position(i,d)) ...
%                               + phi * r3 * (position(exemplar2,d) - position(i,d));
%                 position(i,d) = position(i,d) + velocity(i,d);
% 
%                 % 边界处理 (对应 if() clamp)
%                 if position(i,d) < xmin
%                     position(i,d) = xmin;
%                 elseif position(i,d) > xmax
%                     position(i,d) = xmax;
%                 end
%             end
% 
%             % ---------------- 评价 & FEs 计数 (对应 results[i] = fp->compute(...); FEs++) ----------------
%             FEs = FEs + 1;
%             fitness(i) = fCalculation(position(i,:)', FuncId);
% 
%             % 更新全局最优
%             if fitness(i) < bestever
%                 bestever = fitness(i);
%                 gbestx   = position(i,:);
%             end
% 
%             % 记录历史 (若还有容量)
%             if FEs <= MaxFEs
%                 gbesthistory(FEs) = bestever;
%             end
% 
%             % 检查是否跨越记录点 (对应 C++ 的 record & final_results[counter])
%             if recordIndex <= recordCount && FEs >= recordPoints(recordIndex)
%                 % checkpointBest(recordIndex) = bestever;  % 若要输出可放开
%                 recordIndex = recordIndex + 1;
%             end
% 
%             % 可视化/日志（可选）
%             if  mod(FEs, 1000) == 0
%                 fprintf('RCIPSO Func %d | FEs=%d | best=%e | topo=%d\n', FuncId, FEs, bestever, topology_size);
%             end
%         end
%     end
% end
% 
% % ---------------- 尾部填充 (若 FEs 停在 < MaxFEs) ----------------
% if FEs < MaxFEs
%     gbesthistory(FEs+1:MaxFEs) = bestever;
% end
% end








% function [gbestx, bestever, gbesthistory] = rcipso(mainHandle, popsize, dimension, xmax, xmin, vmax, vmin, maxiter, fCalculation, FuncId, VisualSwitch)
% % RCIPSO  Random Contrastive / Dual-Exemplar Dynamic-Topology PSO (单文件版本)
% % 参照你给出的 HCLPSO 平台函数格式，实现 C++ 代码中拓扑双示例策略。
% %
% % INPUT (为保持平台兼容全部保留；部分参数在此算法中用法说明如下)：
% %   mainHandle    : 可选 GUI/外部句柄（若不为空且存在回调可在内部调用）
% %   popsize       : 种群规模（若留空或 <1 则默认 800）
% %   dimension     : 维度
% %   xmax, xmin    : 标量或向量上下界（长度=dimension 或标量将扩展）
% %   vmax, vmin    : 速度上下界（可选，若为空则不限制或根据区间给出 20% 范围）
% %   maxiter       : 迭代上限(为了接口兼容；该算法以评估次数 MaxFEs 控制，maxiter 可忽略)
% %   fCalculation  : 适应度函数句柄，调用方式 fCalculation(columnVector, FuncId)
% %   FuncId        : 函数编号传递给 fCalculation
% %   VisualSwitch  : 是否打印日志 (true/false)
% %
% % OUTPUT:
% %   gbestx        : 全局最优位置
% %   bestever      : 全局最优适应度
% %   gbesthistory  : 长度 = MaxFEs 的最佳值历史（没有发生新评估的位置填充最近最佳）
% %
% % 核心差异说明：
% %   - 原 C++ 使用 MaxFEs = 3000*dim；本实现保持一致。
% %   - 动态拓扑：topology_size = floor( 2 + (10-2)*sqrt(FEs/MaxFEs) ) 限制在 [2,10]
% %   - 仅当第二优邻居 fitness <= 当前粒子 fitness 时更新该粒子 (并产生一次新的适应度评价 + FEs++)
% %   - exemplar1 = 邻域最优；exemplar2 = 从最差往前找第一个 fitness <= 当前粒子的邻居
% %   - 随机系数 r1,r2,r3 每维重采样（与 C++ for(j) 内调用一致）
% %
% % 使用示例：
% %   [gbestx, bestval, curve] = RCIPSO([], 800, 1000, 100, -100, [], [], [], @myFunc, 3, true);
% %
% % -----------------------------------------------------------------------------
% 
% % ---------------------- 参数检查与默认值 ----------------------
% if nargin < 3 || isempty(dimension),  error('dimension 必须提供'); end
% if nargin < 4 || isempty(xmax),       xmax = 100; end
% if nargin < 5 || isempty(xmin),       xmin = -100; end
% if nargin < 6 || isempty(vmax),       vmax = []; end
% if nargin < 7 || isempty(vmin),       vmin = []; end
% if nargin < 8 || isempty(maxiter),    maxiter = inf; end %#ok<NASGU>
% if nargin < 9 || isempty(fCalculation), error('需提供 fCalculation 句柄'); end
% if nargin <10 || isempty(FuncId),     FuncId = 1; end
% if nargin <11 || isempty(VisualSwitch), VisualSwitch = false; end
% 
% % 标量边界扩展
% if isscalar(xmax), xmax = repmat(xmax,1,dimension); end
% if isscalar(xmin), xmin = repmat(xmin,1,dimension); end
% if numel(xmax) ~= dimension || numel(xmin) ~= dimension
%     error('xmax/xmin 尺寸需为 1 或 dimension');
% end
% 
% % 速度边界 (若未给出则按 20% 区间生成；若给出标量则扩展)
% if isempty(vmax) || isempty(vmin)
%     span = xmax - xmin;
%     vmax = 0.2 * span;
%     vmin = -vmax;
% else
%     if isscalar(vmax), vmax = repmat(vmax,1,dimension); end
%     if isscalar(vmin), vmin = repmat(vmin,1,dimension); end
% end
% 
% % ---------------------- 全局控制 ----------------------
% popsize = 800;
% phi    = 0.3;                 % 与 C++ const double phi = 0.3
% MaxFEs = 3000 * dimension;    % 与 C++ const int MaxFEs = 3000*dim
% FEs    = 0;
% 
% % 记录点
% recordPoints = (5e5:5e5:5e5*10);
% recordCount  = numel(recordPoints);
% recordIndex  = 1;                 % 下一个需要填充的记录点指针
% % checkpointBest = zeros(recordCount,1); % 若后续需要可开放输出
% 
% % ---------------------- 初始化 (位置+速度+初始适应度) ----------------------
% position = xmin + rand(popsize, dimension).*(xmax - xmin);  % 均匀随机
% velocity = zeros(popsize, dimension);
% fitness  = zeros(popsize,1);
% 
% for i = 1:popsize
%     fitness(i) = fCalculation(position(i,:)', FuncId);
% end
% FEs = FEs + popsize;
% 
% [bestever, bestIdx] = min(fitness);
% gbestx = position(bestIdx,:);
% 
% % 预分配历史 (长度 = MaxFEs)。若 FEs 未达到时用 0 占位；后面填充。
% gbesthistory = zeros(MaxFEs,1);
% gbesthistory(1:FEs) = bestever;
% 
% 
% % ---------------------- 主循环 (按 FEs 而非迭代次数) ----------------------
% while FEs <= MaxFEs
%     % 动态拓扑规模
%     rawTopo = 2 + (10 - 2) * sqrt(FEs / MaxFEs);
%     topology_size = floor(rawTopo);
%     if topology_size < 2
%         topology_size = 2;
%     elseif topology_size > 10
%         topology_size = 10;
%     end
% 
%     % 遍历种群
%     for i = 1:popsize
%         if FEs > MaxFEs
%             break;
%         end
% 
%         % ---------- 构造拓扑 (模仿 C++：随机打乱 + 遇到自己跳过再取下一个) ----------
%         perm = randperm(popsize);
%         topo_ids  = zeros(topology_size,1);
%         topo_fits = zeros(topology_size,1);
%         m = 1;
%         for j = 1:topology_size
%             if perm(m) ~= i
%                 topo_ids(j)  = perm(m);
%                 topo_fits(j) = fitness(perm(m));
%                 m = m + 1;
%             else
%                 m = m + 1;
%                 if m > popsize  % 防越界回绕 (C++ 原代码未严谨，此处做保护)
%                     m = 1;
%                     if perm(m) == i
%                         m = m + 1;
%                         if m > popsize
%                             m = 1;
%                         end
%                     end
%                 end
%                 topo_ids(j)  = perm(m);
%                 topo_fits(j) = fitness(perm(m));
%                 m = m + 1;
%                 if m > popsize
%                     m = 1;
%                 end
%             end
%         end
% 
%         % ---------- 排序 ----------
%         [sorted_fits, sortIdx] = sort(topo_fits);
%         sorted_ids = topo_ids(sortIdx);
% 
%         % ---------- 触发条件 ----------
%         if sorted_fits(2) <= fitness(i)
%             exemplar1 = sorted_ids(1);
% 
%             % exemplar2: 从末端往前找 <= fitness(i)
%             jj = topology_size;
%             while jj >= 1 && fitness(sorted_ids(jj)) > fitness(i)
%                 jj = jj - 1;
%             end
%             exemplar2 = sorted_ids(jj);
% 
%             % ---------- 逐维更新 (每维独立 r1,r2,r3) ----------
%             for d = 1:dimension
%                 r1 = rand();
%                 r2 = rand();
%                 r3 = rand();
%                 velocity(i,d) = r1 * velocity(i,d) ...
%                               + r2 * (position(exemplar1,d) - position(i,d)) ...
%                               + phi * r3 * (position(exemplar2,d) - position(i,d));
% 
%                 % 速度边界
%                 if velocity(i,d) > vmax(d), velocity(i,d) = vmax(d); end
%                 if velocity(i,d) < vmin(d), velocity(i,d) = vmin(d); end
% 
%                 position(i,d) = position(i,d) + velocity(i,d);
% 
%                 % 位置边界
%                 if position(i,d) < xmin(d), position(i,d) = xmin(d); end
%                 if position(i,d) > xmax(d), position(i,d) = xmax(d); end
%             end
% 
%             % ---------- 评价 + FEs ----------
%             FEs = FEs + 1;
%             fitness(i) = fCalculation(position(i,:)', FuncId);
% 
%             % 更新全局最优
%             if fitness(i) < bestever
%                 bestever = fitness(i);
%                 gbestx   = position(i,:);
%             end
% 
%             % 写入历史
%             if FEs <= MaxFEs
%                 gbesthistory(FEs) = bestever;
%             end
% 
%             % 记录点检查
%             if recordIndex <= recordCount && FEs >= recordPoints(recordIndex)
%                 % checkpointBest(recordIndex) = bestever; % 若需要可暴露
%                 recordIndex = recordIndex + 1;
%             end
% 
%             % 可视化输出
%             if  mod(FEs, 50000) == 0
%                 fprintf('[RCIPSO] Func=%d FEs=%d gbest=%e topo=%d\n', FuncId, FEs, bestever, topology_size);
%             end
% 
%         end % if condition
%     end % for i
% end % while
% 
% % 尾部填充：若未使用完 MaxFEs，将剩余位置填充为最终 bestever
% if FEs < MaxFEs
%     gbesthistory(FEs+1:MaxFEs) = bestever;
% end
% 
% end % function RCIPSO