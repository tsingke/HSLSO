function [gbestx,bestever,gbesthistory] = DLLSO(mainHandle,popsize,dimension,...
    xmax,xmin,vmax,vmin,maxiter,fCalculation,FuncId,VisualSwitch)

% ==============================================================
% Dynamic Level-Based Learning Swarm Optimizer (DLLSO)
%
% Q. Yang et al.,
% "A Level-Based Learning Swarm Optimizer for Large-Scale Optimization"
% IEEE Transactions on Evolutionary Computation, 2018
%
% 本版本适配当前平台接口：
%
% function [gbestx,bestever,gbesthistory] = DLLSO( ...
%     mainHandle,popsize,dimension,xmax,xmin,vmax,vmin,...
%     maxiter,fCalculation,FuncId,VisualSwitch)
%
% 对应论文 1000-D 实验参数：
% NP  = 500
% phi = 0.4
% S   = {4,6,8,10,20,50}
% MaxFEs = 3e6 = 3000 * 1000
%
% ==============================================================


%% ===================== 参数设置 ===============================

% 论文 1000-D 设置
m = 500;                       % NP
phi = 0.4;                     % 控制第二 exemplar 的影响

% 动态层数候选集合
S = [4, 6, 8, 10, 20, 50];
numS = length(S);

% Eq. (8) 中每个候选层数对应的性能记录
% 论文规定初始化全部为 1
R = ones(1,numS);

ComputeFitness = fCalculation;

FEs = 0;

% 1000维 CEC2010 / CEC2013 主实验评价次数
MaxFEs = 3e6;

% 如果以后想严格按照 3000*D：
% MaxFEs = 3000 * dimension;


%% ===================== 初始化种群 =============================

p = zeros(m,dimension);
v = zeros(m,dimension);
fitness = zeros(1,m);

for i = 1:m

    p(i,:) = xmin + (xmax-xmin).*rand(1,dimension);

    v(i,:) = vmin + (vmax-vmin).*rand(1,dimension);

    fitness(i) = ComputeFitness(p(i,:)',FuncId);

end

FEs = FEs + m;


%% ===================== 初始化全局最优 ==========================

[bestever,id] = min(fitness);

gbestx = p(id,:);


%% ===================== 收敛历史 ===============================

gbesthistory = zeros(MaxFEs,1);

% 和你当前 CSO 平台保持一致：
% 初始化种群的 m 次评价统一记录初始化后的最优值
gbesthistory(1:FEs) = bestever;


gen = 1;


%% ==============================================================
%                      DLLSO 主循环
% ==============================================================

while FEs < MaxFEs


    %% ----------------------------------------------------------
    % 1. 根据 Eq. (9) 计算每个 NL 的选择概率
    %
    %             exp(7*r_i)
    % p_i = -----------------------
    %        sum_j exp(7*r_j)
    %
    % -----------------------------------------------------------

    prob = exp(7 .* R);

    prob = prob ./ sum(prob);


    %% ----------------------------------------------------------
    % 2. Roulette Wheel Selection
    %    动态选择当前这一代的层数 NL
    % -----------------------------------------------------------

    cumulativeProb = cumsum(prob);

    randValue = rand;

    selectedIndex = find(randValue <= cumulativeProb,1,'first');

    NL = S(selectedIndex);


    %% ----------------------------------------------------------
    % 记录本代更新前的 global best
    % 用于 Eq. (8)
    % -----------------------------------------------------------

    oldBest = bestever;


    %% ----------------------------------------------------------
    % 3. 按 fitness 升序排列粒子
    %
    % fitness 越小越好
    %
    % L1：最好
    % L2：次好
    % ...
    % LNL：最差
    %
    % -----------------------------------------------------------

    [~,rankIndex] = sort(fitness,'ascend');


    %% ----------------------------------------------------------
    % 4. 将种群划分为 NL 个层
    %
    % 论文：
    % LS = floor(NP/NL)
    %
    % 如果不能整除，剩余粒子全部加入最低层 LNL
    %
    % -----------------------------------------------------------

    LS = floor(m/NL);

    levels = cell(1,NL);

    for level = 1:NL-1

        startIndex = (level-1)*LS + 1;

        endIndex = level*LS;

        levels{level} = rankIndex(startIndex:endIndex);

    end

    % 最后一层包含所有剩余粒子
    startIndex = (NL-1)*LS + 1;

    levels{NL} = rankIndex(startIndex:m);


    %% ==========================================================
    % 5. 更新 LNL, LNL-1, ..., L3
    %
    % Algorithm 1:
    %
    % for i = NL,...,3
    %
    % 从前 i-1 个更高层中随机选择两个不同的层：
    %
    % L_rl1 和 L_rl2
    %
    % 且：
    %
    % rl1 < rl2 < i
    %
    % 分别从两个层中随机选择一个 exemplar。
    %
    % ==========================================================

    stopFlag = false;

    for level = NL:-1:3

        currentParticles = levels{level};

        numCurrent = length(currentParticles);


        for jj = 1:numCurrent

            currentID = currentParticles(jj);


            %% --------------------------------------------------
            % 从更高的 level 中随机选择两个不同的层
            % ---------------------------------------------------

            selectedLevels = randperm(level-1,2);

            rl1 = selectedLevels(1);
            rl2 = selectedLevels(2);


            % 保证 rl1 < rl2
            %
            % level index 越小，level 越高
            %
            % 因此：
            %
            % exemplar1 来自更好的层
            % exemplar2 来自相对较差的高层
            %

            if rl2 < rl1

                temp = rl1;
                rl1 = rl2;
                rl2 = temp;

            end


            %% --------------------------------------------------
            % 分别从 L_rl1 和 L_rl2 随机选择一个粒子
            % ---------------------------------------------------

            group1 = levels{rl1};
            group2 = levels{rl2};

            k1 = randi(length(group1));
            k2 = randi(length(group2));

            exemplar1 = group1(k1);
            exemplar2 = group2(k2);


            %% --------------------------------------------------
            % Eq. (6)
            %
            % v_i =
            % r1*v_i
            % + r2*(X_rl1,k1 - X_i)
            % + phi*r3*(X_rl2,k2 - X_i)
            %
            % 注意：
            % r1、r2、r3 都是 dimension 维随机向量
            %
            % ---------------------------------------------------

            r1 = rand(1,dimension);
            r2 = rand(1,dimension);
            r3 = rand(1,dimension);

            v(currentID,:) = ...
                r1 .* v(currentID,:) ...
                + r2 .* (p(exemplar1,:) - p(currentID,:)) ...
                + phi .* r3 .* (p(exemplar2,:) - p(currentID,:));


            %% --------------------------------------------------
            % Eq. (7)
            % ---------------------------------------------------

            p(currentID,:) = p(currentID,:) + v(currentID,:);


            %% --------------------------------------------------
            % 边界处理
            %
            % 这里沿用你的 CSO 平台处理方式
            % ---------------------------------------------------

            p(currentID,:) = max(p(currentID,:),xmin);
            p(currentID,:) = min(p(currentID,:),xmax);


            %% --------------------------------------------------
            % Fitness Evaluation
            % ---------------------------------------------------

            fitness(currentID) = ...
                ComputeFitness(p(currentID,:)',FuncId);

            FEs = FEs + 1;


            %% --------------------------------------------------
            % 更新 global best
            % ---------------------------------------------------

            if fitness(currentID) < bestever

                bestever = fitness(currentID);

                gbestx = p(currentID,:);

            end


            gbesthistory(FEs) = bestever;


            %% 每 1000 次评价输出一次
            if mod(FEs,1000) == 0

                fprintf(['DLLSO算法,第%d次评价，' ...
                    '最佳适应度 = %e，NL = %d\n'],...
                    FEs,bestever,NL);

            end


            %% --------------------------------------------------
            % 达到最大评价次数立即结束
            % ---------------------------------------------------

            if FEs >= MaxFEs

                stopFlag = true;

                break;

            end

        end


        if stopFlag
            break;
        end

    end


    if stopFlag

        break;

    end


    %% ==========================================================
    % 6. 单独更新第二层 L2
    %
    % Algorithm 1 Lines 22-32
    %
    % L2 没有两个不同的 higher levels 可以选，
    % 所以：
    %
    % 两个 exemplar 都从 L1 中随机选取。
    %
    % fitness 更好的那个作为 exemplar1，
    % 较差的作为 exemplar2。
    %
    % ==========================================================

    level2Particles = levels{2};

    level1Particles = levels{1};


    for jj = 1:length(level2Particles)

        currentID = level2Particles(jj);


        %% ------------------------------------------------------
        % 从第一层随机选择两个不同粒子
        % -------------------------------------------------------

        selectedParticles = randperm(length(level1Particles),2);

        exemplar1 = level1Particles(selectedParticles(1));
        exemplar2 = level1Particles(selectedParticles(2));


        %% ------------------------------------------------------
        % Algorithm 1:
        %
        % 更好的粒子作为 X_1,k1
        % 较差粒子作为 X_1,k2
        % -------------------------------------------------------

        if fitness(exemplar2) < fitness(exemplar1)

            temp = exemplar1;
            exemplar1 = exemplar2;
            exemplar2 = temp;

        end


        %% ------------------------------------------------------
        % Eq. (6)
        % -------------------------------------------------------

        r1 = rand(1,dimension);
        r2 = rand(1,dimension);
        r3 = rand(1,dimension);

        v(currentID,:) = ...
            r1 .* v(currentID,:) ...
            + r2 .* (p(exemplar1,:) - p(currentID,:)) ...
            + phi .* r3 .* (p(exemplar2,:) - p(currentID,:));


        %% Eq. (7)

        p(currentID,:) = p(currentID,:) + v(currentID,:);


        %% 与平台统一的边界处理

        p(currentID,:) = max(p(currentID,:),xmin);
        p(currentID,:) = min(p(currentID,:),xmax);


        %% Fitness Evaluation

        fitness(currentID) = ...
            ComputeFitness(p(currentID,:)',FuncId);

        FEs = FEs + 1;


        %% global best

        if fitness(currentID) < bestever

            bestever = fitness(currentID);

            gbestx = p(currentID,:);

        end


        gbesthistory(FEs) = bestever;


        if mod(FEs,1000) == 0

            fprintf(['DLLSO算法,第%d次评价，' ...
                '最佳适应度 = %e，NL = %d\n'],...
                FEs,bestever,NL);

        end


        if FEs >= MaxFEs

            stopFlag = true;

            break;

        end

    end


    %% ==========================================================
    % 注意：
    %
    % 第一层 L1 完全不更新。
    %
    % 根据论文：
    %
    % "particles in the first level directly enter
    %  the next generation"
    %
    % 因此这里没有 L1 更新代码。
    % ==========================================================


    %% ==========================================================
    % 7. Eq. (8)
    %
    % 更新刚才被选择的 NL 对应的性能记录 r_i
    %
    %                 |F_old - F_new|
    % r_i = --------------------------------
    %                       |F_old|
    %
    % 其他候选 NL 的记录保持不变。
    %
    % ==========================================================

    if oldBest ~= 0

        R(selectedIndex) = ...
            abs(oldBest-bestever) / abs(oldBest);

    else

        % 已达到 0 时无法继续计算相对改善率。
        % 防止 0/0 导致 NaN。
        R(selectedIndex) = 0;

    end


    gen = gen + 1;


    if stopFlag
        break;
    end

end


%% ==============================================================
% 收敛历史长度处理
% ==============================================================

if FEs < MaxFEs

    gbesthistory(FEs+1:MaxFEs) = bestever;

elseif FEs > MaxFEs

    gbesthistory(MaxFEs+1:end) = [];

end


end