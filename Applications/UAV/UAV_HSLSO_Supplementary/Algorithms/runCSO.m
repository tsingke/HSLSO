function [bestPath, gbesthistory] = runCSO(Fitness, popsize, dimension, xmax, xmin, maxiter)
    % 1. Parameter initialization
    m = popsize; % Use the swarm size passed in, consistently
    phi = 0.1;
    
    % Adaptive velocity bounds
    vmax = (xmax - xmin) * 0.2;
    vmin = -vmax;
    
    FEs = 0;
    MaxFEs = popsize * maxiter;
    
    % 2. Initialize the population and velocity
    p = xmin + (xmax - xmin) .* rand(m, dimension);
    v = vmin + (vmax - vmin) .* rand(m, dimension);

    fitness = zeros(m, 1);
    
    % Initial evaluation
    for i = 1:m
        fitness(i) = Fitness(p(i,:)');
    end
    FEs = FEs + m;
    
    [bestever, id] = min(fitness);
    gbestx = p(id, :);
    
    % Initialize the history record (mapped strictly to length maxiter to align the plots)
    gbesthistory = zeros(1, maxiter);
    gbesthistory(1) = bestever;
    
    % 3. Main loop
    while FEs < MaxFEs
        % Build the competition pairs at random
        rlist = randperm(m);
        % Use floor to handle a possible odd popsize
        half_m = floor(m / 2);
        rpairs = [rlist(1:half_m); rlist(half_m + 1 : 2 * half_m)]';
        
        % Compute the current population center
        center = mean(p);
        
        % Pairwise particle competition
        mask = (fitness(rpairs(:, 1)) > fitness(rpairs(:, 2)));
        for k = 1:half_m
            if mask(k) == 0
                los = rpairs(k, 2);
                win = rpairs(k, 1);
            else
                los = rpairs(k, 1);
                win = rpairs(k, 2);
            end
            
            % The loser learns from the winner and the center position
            v(los, :) = rand(1, dimension) .* v(los, :) + ...
                        rand(1, dimension) .* (p(win, :) - p(los, :)) + ...
                        phi * rand(1, dimension) .* (center - p(los, :));
            p(los, :) = p(los, :) + v(los, :);
            
            % Boundary constraint
            p(los, :) = max(min(p(los, :), xmax), xmin);
            
            % Fitness evaluation (only the loser needs to be re-evaluated)
            fitness(los) = Fitness(p(los, :)');
            FEs = FEs + 1;
            
            % Update the global best
            if fitness(los) < bestever
                bestever = fitness(los);
                gbestx = p(los, :);
            end
            
            % Map FEs exactly to the standard generation index to keep the plots aligned
            idx = min(maxiter, floor(FEs / popsize) + 1);
            gbesthistory(idx) = bestever;
            
            if FEs >= MaxFEs
                break;
            end
        end
    end
    
    % Fill in the history data (prevents trailing zeros when the loop exits early)
    for i = 2:maxiter
        if gbesthistory(i) == 0
            gbesthistory(i) = gbesthistory(i-1);
        end
    end
    
    % Return the best path in 3-D format
   bestPath = decodePath(gbestx);
end
