function [bestPath, gbesthistory] = runSCLDPSO(Fitness, popsize, dimension, xmax, xmin, maxiter)
    % Parameter initialization
    N = 10;
    % Ensure popsize is divisible by N to avoid an index out of range
    if mod(popsize, N) ~= 0
        m = floor(popsize / N);
        popsize = m * N;
    else
        m = popsize / N;
    end

    % Compute the velocity bounds dynamically
    vmax = (xmax - xmin) * 0.2;
    vmin = -vmax;
    
    phi = 0.3;
    FEs = 0;
    MaxFEs = popsize * maxiter;
    
    % Initialize the population and velocity
    p = xmin + (xmax - xmin) .* rand(popsize, dimension);
    v = vmin + (vmax - vmin) .* rand(popsize, dimension);
    fitness = zeros(popsize, 1);
    
    % Initial evaluation
    for i = 1:popsize
        fitness(i) = Fitness(p(i, :)');
    end
    FEs = FEs + popsize;
    
    pbest = p;
    pbestfitness = fitness;
    [bestever, id] = min(fitness);
    gbestx = p(id, :);
    
    % Unify the history record dimension to maxiter
    gbesthistory = zeros(1, maxiter);
    iter = 1;
    gbesthistory(iter) = bestever;
    
    % Main loop
    while FEs < MaxFEs
        iter = iter + 1;
        if iter > maxiter; break; end % safeguard
        
        % Sorting and layering
        [fitness, rank] = sort(fitness);
        p = p(rank, :);
        v = v(rank, :);
        pbest = pbest(rank, :);
        pbestfitness = pbestfitness(rank);
        
        for sub = 1:N
            if sub >= 2
                for k = 1:m
                    i = (sub - 1) * m + k;
                    
                    % Learning exemplar selection (vectorized indexing for speed)
                    if sub == 2
                        learna = pbest(randperm(m, 1), :);
                        learnb = pbest(randperm(m, 1), :);
                    else
                        r = randperm(sub - 1, 2);
                        a = min(r);
                        b = max(r);
                        learna = pbest((a - 1) * m + randperm(m, 1), :);
                        learnb = pbest((b - 1) * m + randperm(m, 1), :);
                    end
                    
                    % Position and velocity update
                    v(i, :) = rand(1, dimension) .* v(i, :) + rand(1, dimension) .* (learna - p(i, :)) + phi .* rand(1, dimension) .* (learnb - p(i, :));
                    p(i, :) = p(i, :) + v(i, :);
                    
                    % Boundary constraint
                    p(i, :) = max(min(p(i, :), xmax), xmin);
                    
                    % Fitness evaluation
                    fitness(i) = Fitness(p(i, :)');
                    FEs = FEs + 1;
                    
                    % Update the personal best and the global best
                    if fitness(i) < pbestfitness(i)
                        pbestfitness(i) = fitness(i);
                        pbest(i, :) = p(i, :);
                    end
                    if fitness(i) < bestever
                        bestever = fitness(i);
                        gbestx = p(i, :);
                    end
                    
                    if FEs >= MaxFEs
                        break;
                    end
                end
            end
            if FEs >= MaxFEs
                break;
            end
        end
        % Record the global best of each generation
        gbesthistory(iter) = bestever;
    end
    
    % Fill in the incomplete iteration history (fixes the undefined gbestfitness error of the original code)
    if iter < maxiter
        gbesthistory(iter+1:maxiter) = bestever;
    elseif iter > maxiter
        gbesthistory(maxiter+1:end) = [];
    end
    
    % Convert the output format to a 3-D waypoint matrix
    bestPath = decodePath(gbestx);
end
