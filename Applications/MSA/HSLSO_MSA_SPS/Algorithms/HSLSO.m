function [gbestx,bestever,gbesthistory]=HSLSO(sequence,aa,lengthdata,L,maxiter,dimension)
% (mainHandle,popsize,dimension,xmax,xmin,vmax,vmin,maxiter,fCalculation, FuncId,VisualSwitch)

popsize=50;
xmax=1;
xmin=0;
vmax = xmax;
vmin = xmin;
N = 10;
m = popsize/N;
phi = 0.3;
PLinit = (1:N)/N;
PLfina = 1-PLinit;

for i =1:popsize
    p(i,:)=xmin+(xmax-xmin).*rand(1,dimension);
    v(i,:) = vmin+(vmax-vmin).*rand(1,dimension);
    fitnessx(i)= fitness(sequence,aa,lengthdata,L,p(i,:)); % ComputeFitness(p(i,:)',FuncId); % 个体适应度    
end
pbest = p;
pbestfitness = fitnessx;
[bestever,id] = max(fitnessx);
gbestx = p(id,:); 
gbesthistory=rand(maxiter,1);
l = 1;

while l<=maxiter
    [fitnessx rank] = sort(fitnessx,'descend');
    p = p(rank,:);
    v = v(rank,:);
    pbest = pbest(rank,:);
    pbestfitness = pbestfitness(rank);

    for sub = 1:N
        if sub >= 2
            PL(sub) = PLinit(sub)+(PLfina(sub)-PLinit(sub))*(l/maxiter);
            for k = 1:m
                if rand<PL(sub)
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
                    fitnessx(i) = fitness(sequence,aa,lengthdata,L,p(i,:));
                    if  fitnessx(i) > pbestfitness(i)
                        pbestfitness(i) = fitnessx(i);
                        pbest(i,:) = p(i,:);
                    end
                    if  fitnessx(i) > bestever
                        bestever = fitnessx(i);
                        gbestx = p(i,:);
                    end
                end
            end
        end
    end
    gbesthistory(l) = bestever;
    fprintf("HSLSO1算法,第%d次评价，最佳适应度 = %e\n",l,bestever);
    l = l+1;
end
end