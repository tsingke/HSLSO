function [gbestx,bestever,gbesthistory]=SCLDPSO(mainHandle,popsize,dimension,xmax,xmin,vmax,vmin,maxiter,fCalculation, FuncId,VisualSwitch)
% 大规模优化的优势组合学习分布式粒子群优化(2023)
popsize = 400;
N = 10;
m = popsize/N;
ComputeFitness = fCalculation;
phi = 0.3;
FEs = 0;
MaxFEs = 3e6;

for i =1:popsize
    p(i,:)=xmin+(xmax-xmin).*rand(1,dimension);
    v(i,:) = vmin+(vmax-vmin).*rand(1,dimension);
    fitness(i)= ComputeFitness(p(i,:)',FuncId); % 个体适应度    
end
FEs = FEs+popsize;
pbest = p;
pbestfitness = fitness;
[bestever,id] = min(fitness);
gbestx = p(id,:); 
gbesthistory = [bestever*ones((FEs),1)];
l = 1;

while FEs < MaxFEs
    [fitness rank] = sort(fitness);
    p = p(rank,:);
    v = v(rank,:);
    pbest = pbest(rank,:);
    pbestfitness = pbestfitness(rank);

    for sub = 1:N
        if sub >= 2
            for k = 1:m
                i = (sub-1)*m+k;
                if sub == 2
                    for j = 1:dimension
                        learna(j) = pbest(randperm(m,1),j);
                        learnb(j) = pbest(randperm(m,1),j);
                    end
                else
                    r = randperm(sub-1,2);
                    if r(1)<r(2)
                        a = r(1);
                        b = r(2);
                    else
                        a = r(2);
                        b = r(1);
                    end
                    for j = 1:dimension
                        learna(j) = pbest((a-1)*m+randperm(m,1),j);
                        learnb(j) = pbest((b-1)*m+randperm(m,1),j);
                    end
                end
                v(i,:) = rand(1,dimension).*v(i,:)+rand(1,dimension).*(learna-p(i,:))+phi.*rand(1,dimension).*(learnb-p(i,:));
                p(i,:) = p(i,:) + v(i,:);
                p(i,:) = max(p(i,:), xmin);
                p(i,:) = min(p(i,:), xmax);
                fitness(i) = ComputeFitness(p(i,:)', FuncId);
                FEs = FEs+1;
                if  fitness(i) < pbestfitness(i)
                    pbestfitness(i) = fitness(i);
                    pbest(i,:) = p(i,:);
                end
                if  fitness(i) < bestever
                    bestever = fitness(i);
                    gbestx = p(i,:);
                end
        
                gbesthistory(FEs) = bestever;
                if mod(FEs, floor(MaxFEs/10)) == 0 && FEs <= MaxFEs
                    fprintf("SCLDPSO算法,第%d次评价，最佳适应度 = %e\n",FEs,bestever);
                end
            end
        end
    end
    l = l+1;
end
    if FEs<MaxFEs
        gbesthistory(FEs+1:MaxFEs)=gbestfitness;
    else
        if FEs>MaxFEs
            gbesthistory(MaxFEs+1:end)=[];
        end
    end
end