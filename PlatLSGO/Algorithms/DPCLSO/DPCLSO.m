function [gbestx,bestever,gbesthistory] = DPCLSO(mainHandle,popsize,...
    dimension,xmax,xmin,vmax,vmin,maxiter,fCalculation,FuncId,VisualSwitch)


ComputeFitness = fCalculation;

%% ================= 参数 ======================================

m = 600;

tau   = 0.01;
alpha = 0.1;
beta  = 0.6;
psi   = 0.3;
k     = 1.0;

MaxFEs = 3e6;
FEs = 0;

%% ================= 初始化 ====================================

p = zeros(m,dimension);

% --------------------------------------------------------------
% 论文没有说明 V(0) 如何初始化。
%
% Algorithm 1 只明确说随机初始化 population in solution space。
% 为避免大随机初速度破坏初期搜索，这里初始化为 0。
% --------------------------------------------------------------
v = zeros(m,dimension);

fitness = zeros(1,m);

for i = 1:m

    p(i,:) = xmin + (xmax-xmin).*rand(1,dimension);

    fitness(i) = ComputeFitness(p(i,:)',FuncId);

end

FEs = FEs + m;

%% ================= 初始化最优 ================================

[bestever,id] = min(fitness);

gbestx = p(id,:);

gbesthistory = zeros(MaxFEs,1);

gbesthistory(1:FEs) = bestever;

gen = 1;

%% ==============================================================
% Main loop
% ==============================================================

while FEs < MaxFEs

    %% ==========================================================
    % 1. Sort particles from best to worst
    %
    % 与 Algorithm 1 Line 4 对应
    %
    % 注意：
    % 这里直接重排 population，
    % 因而排序后：
    %
    % p(1,:) = 当前最好
    % p(2,:) = 当前第二好
    % ...
    % p(m,:) = 当前最差
    %
    % 此时 particle index 本身就是 rank。
    % ==========================================================

    [fitness,order] = sort(fitness,'ascend');

    p = p(order,:);
    v = v(order,:);


    %% ==========================================================
    % 2. Dynamic BPS
    %
    % BPSmin = alpha * PS
    % BPSmax = beta  * PS
    %
    % 论文 Eq.(5) 的排版为：
    %
    % BPSmax - round(BPSmax-BPSmin)*(FE/FEmax)^k
    %
    % 但正文又说 round 用于把计算结果取整。
    %
    % BPS 本身定义为 "number of better particles"，
    % 因此这里将动态减少量取整，使 BPS 始终为整数。
    % ==========================================================

    BPSmin = round(alpha*m);
    BPSmax = round(beta*m);

    BPS = BPSmax - ...
        round((BPSmax-BPSmin)*(FEs/MaxFEs)^k);

    % 安全范围
    BPS = max(BPSmin,min(BPSmax,BPS));


    %% ==========================================================
    % 3. Update probability
    %
    % Eq.(2)
    %
    % pro(i) = 1/[1+exp(-tau*(rank(i)-BPS))]
    %
    % 当前 population 已排序，所以 rank(i)=i。
    % ==========================================================

    rankVector = 1:m;

    pro = 1 ./ ...
        (1 + exp(-tau*(rankVector-BPS)));


    %% ==========================================================
    % 4. Particle update
    %
    % Algorithm 1:
    %
    % for each particle i (rank(i)>2)
    %
    % 因此最好两个粒子不更新。
    %
    % 这里按照排序后的顺序逐个原地更新，
    % 与 Algorithm 1 的执行结构一致。
    % ==========================================================

    for i = 3:m

        if FEs >= MaxFEs
            break;
        end

        %% ------------------------------------------------------
        % Probability controlled update
        % -------------------------------------------------------

        if rand < pro(i)

            %% ==================================================
            % Better particle
            %
            % Algorithm:
            % rank(i) < BPS
            %
            % 两个 learning samples 均来自 better particles，
            % 并且优于 particle i。
            % ==================================================

            if i < BPS

                % ----------------------------------------------
                % ranks 1:i-1 全部优于当前 particle i，
                % 同时它们也都属于 better group。
                % ----------------------------------------------

                candidate = 1:i-1;

                rr = randperm(length(candidate),2);

                sample1 = candidate(rr(1));
                sample2 = candidate(rr(2));


                % ==============================================
                % Eq.(8)
                %
                % IMPORTANT:
                %
                % 对每个维度独立产生随机数。
                %
                % 这是高维 PSO/LSO 实现中更合理的解释，
                % 而不是整个 1000D 共用一个随机标量。
                % ==============================================

                r1 = rand(1,dimension);
                r2 = rand(1,dimension);
                r3 = rand(1,dimension);


                v(i,:) = ...
                    r1 .* v(i,:) ...
                    + r2 .* (p(sample1,:) - p(i,:)) ...
                    + psi .* r3 .* ...
                    (p(sample2,:) - p(i,:));


            %% ==================================================
            % Worse particle
            %
            % 一个 learning sample 来自 better group，
            % 另一个来自 worse group。
            %
            % Algorithm 1 Lines 12-14.
            % ==================================================

            else

                %% ---------------------------------------------
                % learning sample 1:
                % better particle
                %
                % 论文用 rank < BPS 判定 better。
                % 因此 better ranks 为：
                %
                % 1,...,BPS-1
                % ----------------------------------------------

                betterEnd = BPS - 1;

                if betterEnd < 1
                    continue;
                end

                sample1 = randi(betterEnd);


                %% ---------------------------------------------
                % learning sample 2:
                % worse particle
                %
                % 论文正文在这里有歧义：
                % 如果要求该 worse particle 必须优于 i，
                % 那么第一个 worse particle 无解。
                %
                % Algorithm 1 仅要求 "from worse particles"。
                %
                % 因此这里从整个 worse group 中随机选择，
                % 但不能选 particle i 自己。
                % ----------------------------------------------

                worseStart = BPS;

                worseCandidate = worseStart:m;

                worseCandidate(worseCandidate == i) = [];

                if isempty(worseCandidate)
                    continue;
                end

                sample2 = worseCandidate( ...
                    randi(length(worseCandidate)) );


                %% =============================================
                % Eq.(6)
                % ==============================================

                r1 = rand(1,dimension);
                r2 = rand(1,dimension);
                r3 = rand(1,dimension);


                v(i,:) = ...
                    r1 .* v(i,:) ...
                    + r2 .* (p(sample1,:) - p(i,:)) ...
                    + psi .* r3 .* ...
                    (p(sample2,:) - p(i,:));

            end


            %% ==================================================
            % Eq.(7) / Eq.(9)
            %
            % X_i(t+1)=X_i(t)+V_i(t+1)
            % ==================================================

            p(i,:) = p(i,:) + v(i,:);


            %% ==================================================
            % Boundary handling
            %
            % 论文没有明确给出边界策略。
            %
            % 这里与现有实验框架保持一致：
            % saturation / clipping
            % ==================================================

            p(i,:) = max(p(i,:),xmin);
            p(i,:) = min(p(i,:),xmax);


            %% ==================================================
            % Fitness evaluation
            % ==================================================

            fitness(i) = ComputeFitness(p(i,:)',FuncId);

            FEs = FEs + 1;


            %% ==================================================
            % Best-so-far
            % ==================================================

            if fitness(i) < bestever

                bestever = fitness(i);
                gbestx = p(i,:);

            end


            %% ==================================================
            % History
            % ==================================================

            if FEs <= MaxFEs
                gbesthistory(FEs) = bestever;
            end


            %% ==================================================
            % Print every 10%
            % ==================================================

            if mod(FEs, floor(MaxFEs/10)) == 0 && FEs <= MaxFEs

                fprintf(...
                    'DPCLSO算法,第%d次评价，最佳适应度 = %e\n',...
                    FEs,bestever);

            end

        end

    end

    gen = gen + 1;

end


%% ==============================================================
% history 补齐
% ==============================================================

if FEs < MaxFEs

    gbesthistory(FEs+1:MaxFEs) = bestever;

elseif FEs > MaxFEs

    gbesthistory(MaxFEs+1:end) = [];

end

end