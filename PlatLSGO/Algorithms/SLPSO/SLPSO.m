function [bestp,bestever,gbesthistory]=SLPSO(popsize,d,xmax,xmin,vmax,vmin,maxiter,Func,FuncId)
maxfe = 3e6;
lu = [xmin * ones(1, d); xmax * ones(1, d)];
M = 100;
m = M + floor(d/10);% Swarm size is adaptive; the popsize argument passed in is not used
c3 = d/M*0.01;
PL = zeros(m,1);
for i = 1 : m
    PL(i) = (1-(i-1)/m)^log(sqrt(ceil(d/M))); % Learning probability; here i is the rank of the particle fitness value
end
% initialization
XRRmin = repmat(lu(1, :), m, 1);
XRRmax = repmat(lu(2, :), m, 1);
rand('seed', sum(100 * clock));
p = XRRmin + (XRRmax - XRRmin) .* rand(m, d); % Random initialization
for i = 1:m
    fitness(i,:) = Func(p(i,:)', FuncId); 
end
v = zeros(m,d);
bestever = 1e200;
FES = m;

gbesthistory(1:FES) = min(fitness);

%% main loop
while(FES < maxfe)
    [fitness rank] = sort(fitness, 'descend'); % Sort in descending order (worst to best)
    p = p(rank,:);
    v = v(rank,:);
    besty = fitness(m);
    bestp = p(m, :);
    bestever = min(besty, bestever);

    center = mean(p); % Mean position
    randco1 = rand(m, d);
    randco2 = rand(m, d);
    randco3 = rand(m, d);
    winidxmask = repmat([1:m]', [1 d]); % m rows, d columns
    winidx = winidxmask + ceil(rand(m, d).*(m - winidxmask));
    pwin = p;
    for j = 1:d
        pwin(:,j) = p(winidx(:,j),j);
    end
 
    for i = 1:m-1
         if rand<PL(i)
             v(i,:) =  1*(randco1(i,:).*v(i,:) + randco2(i,:).*(pwin(i,:) - p(i,:)) + c3*randco3(i,:).*(center - p(i,:)));
             p(i,:) =  p(i,:) + v(i,:);   

             p(i,:) = max(p(i,:), lu(1,:));
             p(i,:) = min(p(i,:), lu(2,:));
             fitness(i,:) = Func(p(i,:)', FuncId);
             FES = FES + 1;
             % Update Personal Best
             if fitness(i,:)<bestever
                
                bestp=p(i,:);
                bestever=fitness(i,:);
                
             end
             gbesthistory(FES)=bestever;
             if mod(FES, maxfe/1000) == 0 && FES <= maxfe
                 fprintf("SLPSO  FE %d  best = %e\n",FES,bestever);
             end
     end
end

if FES<maxfe
    gbesthistory(FES+1:maxfe)=bestever;
else
    if FES>maxfe
        gbesthistory(maxfe+1:end)=[];
    end
end

end