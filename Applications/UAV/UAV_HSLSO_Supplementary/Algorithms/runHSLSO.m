function [bestPath, gbesthistory] = runHSLSO(Fitness, popsize, dimension, xmax, xmin, maxiter)
    % 1. Parameter initialization
    N = 10; 
    % Ensure popsize is divisible by N to avoid an index out of range
    if mod(popsize, N) ~= 0
        m = floor(popsize / N);
        popsize = m * N;
    else
        m = popsize / N;
    end
    
    % Adaptive velocity bounds
    vmax = (xmax - xmin) * 0.2; 
    vmin = -vmax;
    
    phi = 0.3;
    FEs = 0;
    MaxFEs = popsize * maxiter;
    PLinit = (1:N)/N;
    PLfina = 1 - PLinit;
    
    % Initialize the population and velocity
    p = xmin + (xmax - xmin) .* rand(popsize, dimension);
    v = vmin + (vmax - vmin) .* rand(popsize, dimension);
    fitness = zeros(popsize, 1);
    
    % Initial evaluation
    for i = 1:popsize
        fitness(i) = Fitness(p(i,:)');    
    end
    FEs = FEs + popsize;
    
    pbest = p;
    pbestfitness = fitness;
    [bestever, id] = min(fitness);
    gbestx = p(id, :); 
    
    % Initialize the history record (recorded per generation, aligned with the main framework's plotting)
    gbesthistory = zeros(1, maxiter);
    iter = 1;
    gbesthistory(iter) = bestever;
    
    % 2. Main loop
    while FEs < MaxFEs
        iter = iter + 1;
        if iter > maxiter; break; end % safeguard
        
        % Sorting and layering
        [fitness, rank] = sort(fitness);
        p = p(rank,:);
        v = v(rank,:);
        pbest = pbest(rank,:);
        pbestfitness = pbestfitness(rank);
        
        for sub = 1:N
            if sub >= 2
                PL(sub) = PLinit(sub) + (PLfina(sub) - PLinit(sub)) * (FEs / MaxFEs);
                
                for k = 1:m
                    if rand < PL(sub)
                        i = (sub - 1) * m + k; 
                        
                        % Select the learning exemplars
                        if sub == 2
                            learna = pbest(randperm(m, 1), :);
                            learnb = pbest(randperm(m, 1), :);
                        else
                            k_layers = ceil((sub - 1) * (1 - (FEs / MaxFEs)^2));
                            if k_layers < 2
                                k_layers = 2;
                            end
                            
                            r = randperm(k_layers, 2);
                            a = min(r);
                            b = max(r);
                       
                            learna = pbest((a - 1) * m + randperm(m, 1), :);
                            learnb = pbest((b - 1) * m + randperm(m, 1), :);
                        end
                        
                        % Position update
                        v(i,:) = rand(1,dimension).*v(i,:) + rand(1,dimension).*(learna-p(i,:)) + phi.*rand(1,dimension).*(learnb-p(i,:));
                        p(i,:) = p(i,:) + v(i,:);
                        
                        % Boundary violation handling
                        p(i,:) = max(min(p(i,:), xmax), xmin);
                        
                        % Fitness evaluation
                        fitness(i) = Fitness(p(i,:)');
                        FEs = FEs + 1;
                        
                        % pbest and gbest update
                        if fitness(i) < pbestfitness(i)
                            pbestfitness(i) = fitness(i);
                            pbest(i,:) = p(i,:);
                        end
                        
                        if fitness(i) < bestever
                            bestever = fitness(i);
                            gbestx = p(i,:);
                        end
                        
                        if FEs >= MaxFEs
                            break;
                        end
                    end
                end
            end
            if FEs >= MaxFEs
                break;
            end
        end
        % Record the best of the current generation
        gbesthistory(iter) = bestever;
    end
    
    % Fill in the history (in case the FEs budget ended early)
    if iter < maxiter
        gbesthistory(iter+1:maxiter) = bestever; 
    elseif iter > maxiter
        gbesthistory(maxiter+1:end) = []; 
    end
    
    % Convert the output format to a 3-D waypoint matrix
    bestPath = decodePath(gbestx);
end
