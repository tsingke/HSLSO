function [gbestx,bestever,gbesthistory]=HSLSO(mainHandle,popsize,dimension,xmax,xmin,vmax,vmin,maxiter,fCalculation,FuncId,VisualSwitch)
popsize = 400;
if nargin < 2
    popsize = 400;
end

N = 10;
m = popsize/N;
ComputeFitness = fCalculation;
phi = 0.3;
FEs = 0;
MaxFEs = 3e6;
PLinit = (1:N)/N;
PLfina = 1-PLinit;
baseIndex = (0:dimension-1)*popsize;

p = zeros(popsize,dimension);
v = zeros(popsize,dimension);
fitness = zeros(1,popsize);

for i = 1:popsize
    p(i,:) = xmin+(xmax-xmin).*rand(1,dimension);
    v(i,:) = vmin+(vmax-vmin).*rand(1,dimension);
    fitness(i) = ComputeFitness(p(i,:)',FuncId);
end

FEs = FEs+popsize;
pbest = p;
pbestfitness = fitness;
[bestever,id] = min(fitness);
gbestx = p(id,:);
gbesthistory = bestever*ones(FEs,1);

while FEs < MaxFEs
    [fitness,rank] = sort(fitness);
    p = p(rank,:);
    v = v(rank,:);
    pbest = pbest(rank,:);
    pbestfitness = pbestfitness(rank);

    for sub = 2:N
        PL = PLinit(sub)+(PLfina(sub)-PLinit(sub))*(FEs/MaxFEs);

        for k = 1:m
            if rand < PL
                i = (sub-1)*m+k;

                if sub == 2
                    rowsA = randi(m,1,dimension);
                    rowsB = randi(m,1,dimension);
                    learna = pbest(rowsA+baseIndex);
                    learnb = pbest(rowsB+baseIndex);
                else
                    k_layers = ceil((sub-1)*(1-(FEs/MaxFEs)^2));

                    if k_layers < 2
                        k_layers = 2;
                    end

                    r = randperm(k_layers,2);
                    if r(1) < r(2)
                        a = r(1);
                        b = r(2);
                    else
                        a = r(2);
                        b = r(1);
                    end

                    rowsA = (a-1)*m+randi(m,1,dimension);
                    rowsB = (b-1)*m+randi(m,1,dimension);
                    learna = pbest(rowsA+baseIndex);
                    learnb = pbest(rowsB+baseIndex);
                end

                v(i,:) = rand(1,dimension).*v(i,:)+rand(1,dimension).*(learna-p(i,:))+phi.*rand(1,dimension).*(learnb-p(i,:));
                p(i,:) = p(i,:)+v(i,:);
                p(i,:) = max(p(i,:),xmin);
                p(i,:) = min(p(i,:),xmax);

                fitness(i) = ComputeFitness(p(i,:)',FuncId);
                FEs = FEs+1;

                if fitness(i) < pbestfitness(i)
                    pbestfitness(i) = fitness(i);
                    pbest(i,:) = p(i,:);
                end

                if fitness(i) < bestever
                    bestever = fitness(i);
                    gbestx = p(i,:);
                end

                gbesthistory(FEs) = bestever;
                if mod(FEs, floor(MaxFEs/10)) == 0 && FEs <= MaxFEs
                    fprintf("HSLSO算法,第%d次评价，最佳适应度 = %e\n",FEs,bestever);
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
end

if FEs < MaxFEs
    gbesthistory(FEs+1:MaxFEs) = bestever;
elseif FEs > MaxFEs
    gbesthistory(MaxFEs+1:end) = [];
end
end




