function [gbestx,bestever,gbesthistory]=CSO(mainHandle,popsize,dimension,xmax,xmin,vmax,vmin,maxiter,fCalculation, FuncId,VisualSwitch)
m = 500;
ComputeFitness = fCalculation;
phi = 0.1;
FEs = 0;
MaxFEs = 3e6;
%MaxFEs = 10000 * dimension;

for i =1:m
    p(i,:)=xmin+(xmax-xmin).*rand(1,dimension);
    v(i,:) = vmin+(vmax-vmin).*rand(1,dimension);
    fitness(i)= ComputeFitness(p(i,:)',FuncId); % 个体适应度    
end
FEs = FEs+m;

[bestever,id] = min(fitness);
gbestx = p(id,:); 

gbesthistory = [bestever*ones((FEs),1)]; 

gen = 1;

while FEs < MaxFEs

    % 随机构建竞争对
    rlist = randperm(m);
    rpairs = [rlist(1:ceil(m/2)); rlist(floor(m/2) + 1:m)]';
        
    % 计算中心位置
    center = mean(p);
        
    % 粒子对竞争
    mask = (fitness(rpairs(:,1))>fitness(rpairs(:,2)));
    for k = 1:ceil(m/2)
        if mask(k)==0
            los=rpairs(k,2);
            win=rpairs(k,1);
        else
            los=rpairs(k,1);
            win=rpairs(k,2);
        end
        v(los,:) = rand(1,dimension).*v(los,:)+rand(1,dimension).*(p(win,:)-p(los,:))+ phi*rand(1,dimension).*(center-p(los,:));
        p(los,:) = p(los,:) + v(los,:);
        p(los,:) = max(p(los,:), xmin);
        p(los,:) = min(p(los,:), xmax);
        fitness(los) = ComputeFitness(p(los,:)', FuncId);
        FEs = FEs+1;
        if  fitness(los) < bestever
            bestever = fitness(los);
            gbestx = p(los,:);
        end

        gbesthistory(FEs) = bestever;
        if mod(FEs, floor(MaxFEs/10)) == 0 && FEs <= MaxFEs
           fprintf("CSO算法,第%d次评价，最佳适应度 = %e\n",FEs,bestever);
        end
        
    end

    gen = gen+1;

end

    if FEs<MaxFEs
        gbesthistory(FEs+1:MaxFEs)=gbestfitness;
    else
        if FEs>MaxFEs
            gbesthistory(MaxFEs+1:end)=[];
        end
    end
end