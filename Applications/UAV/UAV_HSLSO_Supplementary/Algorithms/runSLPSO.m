function [bestPath, gbesthistory] = runSLPSO(Fitness, popsize_in, dimension, xmax, xmin, maxiter)
    % 1. Adaptive population size computation of SLPSO
    M = 100;
    m = M + floor(dimension / 10); % The popsize_in argument is not used for initialization; m is used instead
    
    % 2. Maximum budget of the unified framework (ensures a fair comparison with the other algorithms)
    MaxFEs = popsize_in * maxiter; 
    FEs = 0;
    
    c3 = (dimension / M) * 0.01;
    PL = zeros(m, 1);
    for i = 1:m
        PL(i) = (1 - (i - 1) / m)^log(sqrt(ceil(dimension / M))); % Learning probability
    end
    
    % Initialize the population
     p = xmin + (xmax - xmin) .* rand(m, dimension);
    fitness = zeros(m, 1);
    for i = 1:m
        fitness(i) = Fitness(p(i, :)'); 
    end
    FEs = FEs + m;
    
     v = zeros(m, dimension);
    [bestever, best_idx] = min(fitness);
    gbestx = p(best_idx, :);
    
    % Initialize the history record (recorded per generation, aligned with the main framework's plotting)
    gbesthistory = zeros(1, maxiter);
    iter = 1;
    gbesthistory(iter) = bestever;
    
    %% Main loop
    while FEs < MaxFEs
        iter = iter + 1;
        if iter > maxiter; break; end % guard against a plot index out of range
        
        % Sort in descending order (worst to best: rank 1 is the worst, rank m is the best)
        [fitness, rank] = sort(fitness, 'descend');
        p = p(rank, :);
        v = v(rank, :);
        
        % Update the historical best
        current_besty = fitness(m);
        current_bestp = p(m, :);
        if current_besty < bestever
            bestever = current_besty;
            gbestx = current_bestp;
        end
        
        center = mean(p); % Mean position of the current population
        
        randco1 = rand(m, dimension);
        randco2 = rand(m, dimension);
        randco3 = rand(m, dimension);
        
        % Find the learning target (must be ranked higher than itself)
        winidxmask = repmat((1:m)', [1, dimension]);
        winidx = winidxmask + ceil(rand(m, dimension) .* (m - winidxmask));
        
        pwin = zeros(m, dimension);
        for j = 1:dimension
            pwin(:, j) = p(winidx(:, j), j);
        end
        
        % Probabilistic update for all particles except the best one
        for i = 1:m-1
            if rand < PL(i)
                v(i, :) = randco1(i, :) .* v(i, :) + ...
                          randco2(i, :) .* (pwin(i, :) - p(i, :)) + ...
                          c3 * randco3(i, :) .* (center - p(i, :));
                p(i, :) = p(i, :) + v(i, :);
                
                % Boundary constraint
                p(i, :) = max(p(i, :), xmin);
                p(i, :) = min(p(i, :), xmax);
                
                % Fitness evaluation
                fitness(i) = Fitness(p(i, :)');
                FEs = FEs + 1;
                
                if fitness(i) < bestever
                    gbestx = p(i, :);
                    bestever = fitness(i);
                end
                
                if FEs >= MaxFEs
                    break;
                end
            end
        end
        % Record the best of the current generation
        gbesthistory(iter) = bestever;
    end
    
    % Fill in the history data
    if iter < maxiter
        gbesthistory(iter+1:maxiter) = bestever;
    elseif iter > maxiter
        gbesthistory(maxiter+1:end) = [];
    end
    
    % Return the best path in 3-D format
    bestPath = decodePath(gbestx);
end
