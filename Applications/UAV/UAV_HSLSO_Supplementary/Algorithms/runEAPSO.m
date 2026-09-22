function [bestPath, gbesthistory] = runEAPSO(Fitness, popsize, dimension, xmax, xmin, maxiter)
% runEAPSO
% Elite Archive Particle Swarm Optimizer for UAV Path Planning
%
% 输入:
%   Fitness   : 适应度函数句柄，输入列向量
%   popsize   : 种群规模
%   dimension : 决策变量维度
%   xmax      : 上界
%   xmin      : 下界
%   maxiter   : 最大迭代次数
%
% 输出:
%   bestPath      : decodePath 后的最优路径
%   gbesthistory  : 每代最优适应度历史

    %% ===================== 1. 参数初始化 =====================
    FEs = 0;
    MaxFEs = popsize * maxiter;

    % 速度边界，建议不要太大，否则路径规划中容易跳动
    vmax = 0.2 * (xmax - xmin);
    vmin = -vmax;

    % 如果 xmax/xmin 是标量，扩展为行向量
    if isscalar(xmax)
        xmax = xmax * ones(1, dimension);
    end

    if isscalar(xmin)
        xmin = xmin * ones(1, dimension);
    end

    if isscalar(vmax)
        vmax = vmax * ones(1, dimension);
        vmin = -vmax;
    end

    %% ===================== 2. 初始化种群 =====================
    p = xmin + (xmax - xmin) .* rand(popsize, dimension);
    v = vmin + (vmax - vmin) .* rand(popsize, dimension);

    fitness = zeros(popsize, 1);

    for i = 1:popsize
        fitness(i) = Fitness(p(i, :)');
    end

    FEs = FEs + popsize;

    %% ===================== 3. 初始化个体最优和全局最优 =====================
    pbest = p;
    pbestfitness = fitness;

    [bestever, id] = min(fitness);
    gbestx = p(id, :);

    %% ===================== 4. 初始化 PART 和 GART 档案 =====================
    % PART: 历史个体优秀档案
    % GART: 历史全局优秀档案

    [~, sort_id] = sort(pbestfitness);

    PART.Position = [];
    PART.Cost = [];

    GART.Position = [];
    GART.Cost = [];

    % PART 初始放入当前最好的两个个体
    PART(1).Position = pbest(sort_id(1), :);
    PART(1).Cost = pbestfitness(sort_id(1));

    if popsize >= 2
        PART(2).Position = pbest(sort_id(2), :);
        PART(2).Cost = pbestfitness(sort_id(2));
        pcount = 2;
    else
        pcount = 1;
    end

    % GART 初始放入当前全局最优
    GART(1).Position = gbestx;
    GART(1).Cost = bestever;
    gcount = 1;

    % 档案最大容量
    NP = popsize;

    %% ===================== 5. 初始化历史记录 =====================
    gbesthistory = zeros(1, maxiter);

    iter = 1;
    gbesthistory(iter) = bestever;

    %% ===================== 6. 主循环 =====================
    while FEs < MaxFEs

        iter = iter + 1;
        if iter > maxiter
            break;
        end

        %% ---------- 根据个体最优排序 ----------
        [~, ind] = sort(pbestfitness);

        half = floor(popsize / 2);

        if half < 1
            break;
        end

        WINNER = ind(1:half);
        LOSER  = ind(half+1:min(2*half, popsize));

        if isempty(LOSER)
            gbesthistory(iter) = bestever;
            continue;
        end

        loser_cost_mean = mean(pbestfitness(LOSER));

        %% ===================== 7. 更新败者个体 =====================
        for ii = 1:length(LOSER)

            if FEs >= MaxFEs
                break;
            end

            loser_idx = LOSER(ii);

            %% ===== 从 PART 中选择较优历史个体 =====
            if length(PART) > 1
                a = randi(length(PART));
                b = randi(length(PART));

                while a == b
                    b = randi(length(PART));
                end

                if PART(b).Cost < PART(a).Cost
                    a = b;
                end
            else
                a = 1;
            end

            %% ===== 从 GART 中选择较优全局历史个体 =====
            if length(GART) > 1
                c = randi(length(GART));
                d = randi(length(GART));

                while c == d
                    d = randi(length(GART));
                end

                if GART(d).Cost < GART(c).Cost
                    b_gart = d;
                else
                    b_gart = c;
                end
            else
                b_gart = 1;
            end

            %% ===== 从 WINNER 中选择较优胜者 =====
            if length(WINNER) > 1
                c = randi(length(WINNER));
                q = randi(length(WINNER));

                while c == q
                    q = randi(length(WINNER));
                end

                if pbestfitness(WINNER(q)) < pbestfitness(WINNER(c))
                    c = q;
                end
            else
                c = 1;
            end

            winner_idx = WINNER(c);

            %% ===================== 8. EAPSO 核心速度更新 =====================
            w  = rand(1, dimension);
            F1 = rand(1, dimension);
            F2 = rand(1, dimension);

            current_pos = p(loser_idx, :);

            % loser 中相对较好的个体
            if pbestfitness(loser_idx) < loser_cost_mean

                if PART(a).Cost < GART(b_gart).Cost && ...
                   PART(a).Cost < pbestfitness(winner_idx)

                    v(loser_idx, :) = ...
                        w .* v(loser_idx, :) ...
                        + F1 .* (PART(a).Position - current_pos) ...
                        + F2 .* (gbestx - current_pos);

                elseif PART(a).Cost > GART(b_gart).Cost && ...
                       GART(b_gart).Cost < pbestfitness(winner_idx)

                    v(loser_idx, :) = ...
                        w .* v(loser_idx, :) ...
                        + F1 .* (GART(b_gart).Position - current_pos) ...
                        + F2 .* (gbestx - current_pos);

                elseif PART(a).Cost > pbestfitness(winner_idx) && ...
                       GART(b_gart).Cost > pbestfitness(winner_idx)

                    v(loser_idx, :) = ...
                        w .* v(loser_idx, :) ...
                        + F1 .* (pbest(winner_idx, :) - current_pos) ...
                        + F2 .* (gbestx - current_pos);

                else

                    v(loser_idx, :) = ...
                        w .* v(loser_idx, :) ...
                        + F1 .* (gbestx - current_pos);

                end

            % loser 中相对较差的个体
            else

                if PART(a).Cost > GART(b_gart).Cost && ...
                   PART(a).Cost > pbestfitness(winner_idx)

                    v(loser_idx, :) = ...
                        w .* v(loser_idx, :) ...
                        + F1 .* (GART(b_gart).Position - current_pos) ...
                        + F2 .* (pbest(winner_idx, :) - current_pos);

                elseif PART(a).Cost < GART(b_gart).Cost && ...
                       GART(b_gart).Cost > pbestfitness(winner_idx)

                    v(loser_idx, :) = ...
                        w .* v(loser_idx, :) ...
                        + F1 .* (PART(a).Position - current_pos) ...
                        + F2 .* (pbest(winner_idx, :) - current_pos);

                elseif PART(a).Cost < pbestfitness(winner_idx) && ...
                       GART(b_gart).Cost < pbestfitness(winner_idx)

                    v(loser_idx, :) = ...
                        w .* v(loser_idx, :) ...
                        + F1 .* (PART(a).Position - current_pos) ...
                        + F2 .* (GART(b_gart).Position - current_pos);

                else

                    v(loser_idx, :) = ...
                        w .* v(loser_idx, :) ...
                        + F1 .* (pbest(winner_idx, :) - current_pos);

                end
            end

            %% ===================== 9. 速度边界控制 =====================
            v(loser_idx, :) = max(v(loser_idx, :), vmin);
            v(loser_idx, :) = min(v(loser_idx, :), vmax);

            %% ===================== 10. 位置更新 =====================
            p(loser_idx, :) = p(loser_idx, :) + v(loser_idx, :);

            %% ===================== 11. 位置边界控制 =====================
            IsOutside = p(loser_idx, :) < xmin | p(loser_idx, :) > xmax;

            % 越界速度反向
            v(loser_idx, IsOutside) = -v(loser_idx, IsOutside);

            % 位置截断
            p(loser_idx, :) = max(p(loser_idx, :), xmin);
            p(loser_idx, :) = min(p(loser_idx, :), xmax);

            %% ===================== 12. 适应度评价 =====================
            fitness(loser_idx) = Fitness(p(loser_idx, :)');
            FEs = FEs + 1;

            %% ===================== 13. 更新个体最优和 PART =====================
            if fitness(loser_idx) < pbestfitness(loser_idx)

                pbest(loser_idx, :) = p(loser_idx, :);
                pbestfitness(loser_idx) = fitness(loser_idx);

                % 更新 PART 档案
                pcount = pcount + 1;

                if pcount <= NP

                    PART(pcount).Position = p(loser_idx, :);
                    PART(pcount).Cost = fitness(loser_idx);

                else

                    aa = randi(NP);
                    bb = randi(NP);

                    while aa == bb
                        bb = randi(NP);
                    end

                    % 找到较差的档案成员作为替换候选
                    if PART(aa).Cost < PART(bb).Cost
                        replace_idx = bb;
                    else
                        replace_idx = aa;
                    end

                    if fitness(loser_idx) < PART(replace_idx).Cost
                        PART(replace_idx).Position = p(loser_idx, :);
                        PART(replace_idx).Cost = fitness(loser_idx);
                    end
                end
            end

            %% ===================== 14. 更新全局最优 =====================
            if pbestfitness(loser_idx) < bestever
                bestever = pbestfitness(loser_idx);
                gbestx = pbest(loser_idx, :);
            end

            if FEs >= MaxFEs
                break;
            end

        end

        %% ===================== 15. 更新 GART 档案 =====================
        gcount = gcount + 1;

        if gcount <= NP

            GART(gcount).Position = gbestx;
            GART(gcount).Cost = bestever;

        else

            aa = randi(length(GART));
            bb = randi(length(GART));

            while aa == bb
                bb = randi(length(GART));
            end

            % 找到较差的 GART 成员作为替换候选
            if GART(aa).Cost < GART(bb).Cost
                replace_idx = bb;
            else
                replace_idx = aa;
            end

            if bestever < GART(replace_idx).Cost
                GART(replace_idx).Position = gbestx;
                GART(replace_idx).Cost = bestever;
            end
        end

        %% ===================== 16. 记录当代最优 =====================
        gbesthistory(iter) = bestever;

    end

    %% ===================== 17. 补齐历史记录 =====================
    if iter < maxiter
        gbesthistory(iter+1:maxiter) = bestever;
    elseif iter > maxiter
        gbesthistory(maxiter+1:end) = [];
    end

    %% ===================== 18. 输出最优路径 =====================
    bestPath = decodePath(gbestx);

end
