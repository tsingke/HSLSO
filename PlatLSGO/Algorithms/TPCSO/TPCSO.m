function [gbestX,gbestfitness,gbesthistory]=TPCSO(mainHandle,popsize,dimension,xmax,xmin,vmax,vmin,maxiter,fCalculation, FuncId,VisualSwitch)
% 复现版本
popsize = 500;

if FuncId == 13
    phi1 = 0.1; phi2 = 0.1; phi3 = 0.1;
else
    phi1 = 0.3; phi2 = 0.3; phi3 = 0.3;
end

lu = [xmin * ones(1, dimension); xmax * ones(1, dimension)];
ComputeFitness = fCalculation;
gbesthistory=rand(maxiter,1);

FEs = 0;
MaxFEs = 3e6;
T = round(MaxFEs/6);

%% initialization
for i =1:popsize
    p(i,:) = xmin+(xmax-xmin).*rand(1,dimension);
    v(i,:) = vmin+(vmax-vmin).*rand(1,dimension);
    fitness(i)= ComputeFitness(p(i,:)',FuncId); % 个体适应度  
    FEs = FEs+1;
end

[gbestfitness,id] = min(fitness);
gbestX = p(id,:);
gbesthistory = [gbestfitness*ones((FEs),1)];

while FEs < MaxFEs
    % 随机分成两个子种群
    sub1 = p(1:ceil(popsize/2),:); 
    sub2 = p(floor(popsize/2) + 1:popsize,:);

    l1 = size(sub1,1);
    l2 = size(sub2,1);

    % 生成随机粒子对
    rlist1 = randperm(l1);
    rpairs1 = [rlist1(1:ceil(l1/2)); rlist1(floor(l1/2) + 1:l1)]';

    rlist2 = randperm(l2)+l1;
    rpairs2 = [rlist2(1:ceil(l2/2)); rlist2(floor(l2/2) + 1:l2)]';
    
    % calculate the center position
    center1 = mean(sub1);
    center2 = mean(sub2);
    
    % do pairwise competitions
    % 子种群1
    for i = 1:ceil(l1/2)
        if (fitness(rpairs1(i,1))<fitness(rpairs1(i,2)))
            losers1(i) = rpairs1(i,2);
            winners1(i) = rpairs1(i,1);
        else
            losers1(i) = rpairs1(i,1);
            winners1(i) = rpairs1(i,2);
        end
        % 三阶段
        if FEs<=T
            v(losers1(i),:) = rand(1,dimension).*v(losers1(i),:)+rand(1,dimension).*(p(winners1(i),:)-p(losers1(i),:))+phi1*rand(1,dimension).*(center1-p(losers1(i),:));        
        elseif FEs<=(MaxFEs-T)
            v(losers1(i),:) = rand(1,dimension).*v(losers1(i),:)+rand(1,dimension).*(p(winners1(i),:)-p(losers1(i),:))+phi2*rand(1,dimension).*(p(winners2(i),:)-p(losers1(i),:));         
        else
            v(losers1(i),:) = rand(1,dimension).*v(losers1(i),:)+rand(1,dimension).*(p(winners1(i),:)-p(losers1(i),:))+phi3*rand(1,dimension).*(gbestX-p(losers1(i),:));
            
        end
        p(losers1(i),:)=p(losers1(i),:)+v(losers1(i),:);
        % 位置边界控制
        p(losers1(i),:) = max(p(losers1(i),:), (lu(1, :)));
        p(losers1(i),:) = min(p(losers1(i),:), (lu(2, :)));

        fitness(losers1(i)) = ComputeFitness(p(losers1(i),:)',FuncId);
        FEs = FEs+1;

        if fitness(losers1(i))<= gbestfitness
            gbestfitness = fitness(losers1(i));
            gbestX =p(losers1(i),:);
        end
        gbesthistory(FEs) = gbestfitness;
        if mod(FEs, floor(MaxFEs/10)) == 0 && FEs <= MaxFEs
            fprintf("TPCSO算法，第%d次评价，最佳适应度=%e\n",FEs,gbestfitness);
        end
    end

    % 子种群2
    for i = 1:ceil(l2/2)
        if (fitness(rpairs2(i,1))<fitness(rpairs2(i,2)))
            losers2(i) = rpairs2(i,2);
            winners2(i) = rpairs2(i,1);
        else
            losers2(i) = rpairs2(i,1);
            winners2(i) = rpairs2(i,2);
        end
        % 三阶段
        if FEs<=T
            v(losers2(i),:) = rand(1,dimension).*v(losers2(i),:)+rand(1,dimension).*(p(winners2(i),:)-p(losers2(i),:))+phi1*rand(1,dimension).*(center2-p(losers2(i),:));       
        elseif FEs<=(MaxFEs-T)
            v(losers2(i),:) = rand(1,dimension).*v(losers2(i),:)+rand(1,dimension).*(p(winners2(i),:)-p(losers2(i),:))+phi2*rand(1,dimension).*(p(winners1(i),:)-p(losers2(i),:));            
        else
            v(losers2(i),:) = rand(1,dimension).*v(losers2(i),:)+rand(1,dimension).*(p(winners2(i),:)-p(losers2(i),:))+phi3*rand(1,dimension).*(gbestX-p(losers2(i),:));            
        end
        p(losers2(i),:)=p(losers2(i),:)+v(losers2(i),:);
        % 位置边界控制
        p(losers2(i),:) = max(p(losers2(i),:), (lu(1, :)));
        p(losers2(i),:) = min(p(losers2(i),:), (lu(2, :)));
        fitness(losers2(i)) = ComputeFitness(p(losers2(i),:)',FuncId);
        FEs = FEs+1;

        if  fitness(losers2(i))<= gbestfitness
            gbestfitness = fitness(losers2(i));
            gbestX =p(losers2(i),:);
        end
        gbesthistory(FEs) = gbestfitness;
        if mod(FEs, floor(MaxFEs/10)) == 0 && FEs <= MaxFEs
            fprintf("TPCSO算法，第%d次评价，最佳适应度=%e\n",FEs,gbestfitness);
        end
    end

end
    if FEs<MaxFEs
        gbesthistory(FEs+1:MaxFEs)=gbestfitness;
    else
        if FEs>MaxFEs
            gbesthistory(MaxFEs+1:end)=[];
        end
    end         
end