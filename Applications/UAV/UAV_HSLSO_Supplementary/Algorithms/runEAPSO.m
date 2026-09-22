function [bestPath, gbesthistory] = runEAPSO(Fitness, popsize, dimension, xmax, xmin, maxiter)
% runEAPSO
% Elite Archive Particle Swarm Optimizer for UAV Path Planning
%
% INPUTS:
%   Fitness   : fitness function handle, takes a column vector
%   popsize   : swarm size
%   dimension : number of decision variables
%   xmax      : upper bound
%   xmin      : lower bound
%   maxiter   : maximum number of iterations
%
% OUTPUTS:
%   bestPath      : optimal path returned by decodePath
%   gbesthistory  : history of the best fitness per generation

    %% ===================== 1. Parameter initialization =====================
    FEs = 0;
    MaxFEs = popsize * maxiter;

    % Velocity bounds; it is advisable not to make them too large, otherwise the path tends to jump around during planning
    vmax = 0.2 * (xmax - xmin);
    vmin = -vmax;

    % If xmax/xmin is a scalar, expand it to a row vector
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

    %% ===================== 2. Initialize the population =====================
    p = xmin + (xmax - xmin) .* rand(popsize, dimension);
    v = vmin + (vmax - vmin) .* rand(popsize, dimension);

    fitness = zeros(popsize, 1);

    for i = 1:popsize
        fitness(i) = Fitness(p(i, :)');
    end

    FEs = FEs + popsize;

    %% ===================== 3. Initialize the personal best and the global best =====================
    pbest = p;
    pbestfitness = fitness;

    [bestever, id] = min(fitness);
    gbestx = p(id, :);

    %% ===================== 4. Initialize the PART and GART archives =====================
    % PART: archive of historically good personal solutions
    % GART: archive of historically good global solutions

    [~, sort_id] = sort(pbestfitness);

    PART.Position = [];
    PART.Cost = [];

    GART.Position = [];
    GART.Cost = [];

    % Put the two currently best individuals into PART at initialization
    PART(1).Position = pbest(sort_id(1), :);
    PART(1).Cost = pbestfitness(sort_id(1));

    if popsize >= 2
        PART(2).Position = pbest(sort_id(2), :);
        PART(2).Cost = pbestfitness(sort_id(2));
        pcount = 2;
    else
        pcount = 1;
    end

    % Put the current global best into GART at initialization
    GART(1).Position = gbestx;
    GART(1).Cost = bestever;
    gcount = 1;

    % Maximum archive capacity
    NP = popsize;

    %% ===================== 5. Initialize the history record =====================
    gbesthistory = zeros(1, maxiter);

    iter = 1;
    gbesthistory(iter) = bestever;

    %% ===================== 6. Main loop =====================
    while FEs < MaxFEs

        iter = iter + 1;
        if iter > maxiter
            break;
        end

        %% ---------- Sort by the personal best ----------
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

        %% ===================== 7. Update the losing individuals =====================
        for ii = 1:length(LOSER)

            if FEs >= MaxFEs
                break;
            end

            loser_idx = LOSER(ii);

            %% ===== Select the better historical individual from PART =====
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

            %% ===== Select the better global historical individual from GART =====
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

            %% ===== Select the better winner from WINNER =====
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

            %% ===================== 8. EAPSO core velocity update =====================
            w  = rand(1, dimension);
            F1 = rand(1, dimension);
            F2 = rand(1, dimension);

            current_pos = p(loser_idx, :);

            % The relatively better individual among the losers
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

            % The relatively worse individual among the losers
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

            %% ===================== 9. Velocity boundary control =====================
            v(loser_idx, :) = max(v(loser_idx, :), vmin);
            v(loser_idx, :) = min(v(loser_idx, :), vmax);

            %% ===================== 10. Position update =====================
            p(loser_idx, :) = p(loser_idx, :) + v(loser_idx, :);

            %% ===================== 11. Position boundary control =====================
            IsOutside = p(loser_idx, :) < xmin | p(loser_idx, :) > xmax;

            % Reverse the velocity when out of bounds
            v(loser_idx, IsOutside) = -v(loser_idx, IsOutside);

            % Truncate the position
            p(loser_idx, :) = max(p(loser_idx, :), xmin);
            p(loser_idx, :) = min(p(loser_idx, :), xmax);

            %% ===================== 12. Fitness evaluation =====================
            fitness(loser_idx) = Fitness(p(loser_idx, :)');
            FEs = FEs + 1;

            %% ===================== 13. Update the personal best and PART =====================
            if fitness(loser_idx) < pbestfitness(loser_idx)

                pbest(loser_idx, :) = p(loser_idx, :);
                pbestfitness(loser_idx) = fitness(loser_idx);

                % Update the PART archive
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

                    % Find the worse archive member as the replacement candidate
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

            %% ===================== 14. Update the global best =====================
            if pbestfitness(loser_idx) < bestever
                bestever = pbestfitness(loser_idx);
                gbestx = pbest(loser_idx, :);
            end

            if FEs >= MaxFEs
                break;
            end

        end

        %% ===================== 15. Update the GART archive =====================
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

            % Find the worse GART member as the replacement candidate
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

        %% ===================== 16. Record the best of the current generation =====================
        gbesthistory(iter) = bestever;

    end

    %% ===================== 17. Fill in the history record =====================
    if iter < maxiter
        gbesthistory(iter+1:maxiter) = bestever;
    elseif iter > maxiter
        gbesthistory(maxiter+1:end) = [];
    end

    %% ===================== 18. Output the optimal path =====================
    bestPath = decodePath(gbestx);

end
