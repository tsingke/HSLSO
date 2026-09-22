% Author: Dr. Yuan SUN
% email address: yuan.sun@unimelb.edu.au OR suiyuanpku@gmail.com
%
% ------------
% Description:
% ------------
% ccos - This function implements the CC framework that can select an appropriate 
%        optimizer from a given algorith pool for solving large-scale problems.
function [xbest,bestval,gbesthistory] = CCOS(~,NP,dim,xmax,xmin,~,~,~,Func,func_num,~)
%     global gbesthistory;
%     global countFE;

    decResults = sprintf('./Algorithms/CCOS/CEC2013/rdg/results/F%02d', func_num);
    load (decResults);
    FEMax = 3e6; % 优化阶段使用的评估次数等于最大评估次数减分组阶段使用的评估次数

    Lbound = xmin.*ones(NP,dim);
    Ubound = xmax.*ones(NP,dim);

    % trace for the fitness value 跟踪适应度值
    groupingFE = 3e6 - FEs; % FEs used in decomposition
    
    countFE = FEs; % 评估次数
    maxFECycle = 10000; % 每轮循环使用的最大评估次数
  
    % grouping cell 分组元组
    allGroups = grouping(func_num); % 加载RDG方法的分组结果
    numGroups = size(allGroups, 2); % 组数
    
    % initialization for sansde 优化器的初始化
    popsize = NP; % 种群规模，这里NP=100，在run函数中定义
    ccm = 0.5*ones(1,numGroups);
    pop = Lbound + rand(popsize, dim) .* (Ubound-Lbound); % 种群初始化
    for i = 1:popsize
        val(i) = Func(pop(i,:)',func_num);
    end
    [bestval, ibest] = min(val); % 获取最佳适应度值和最佳适应度值个体
    xbest = pop(ibest, :);
    countFE=countFE+popsize; % 评估次数
%     fprintf(fid1, '%d, %e\n', groupingFE, bestval);
    
    % Initialization for slpso
    v = zeros(NP, dim);
    
    % Initialization of fitness improvement array 初始化适应度提升矩阵
    fitImpAccum = zeros(1,2*numGroups);
    fitness = repmat(val', [1 numGroups]);
     
    cycle = 0;
    display = 1; % 1 display results; 0 not display results 
    while (countFE < FEMax)
        cycle = cycle + 1;   
        maxVal = max(fitImpAccum); % 获取适应度提升矩阵的最大值
        idx    = find(fitImpAccum == maxVal); % 获取适应度提升矩阵最大值的索引值
        for i = 1:length(idx)    
            groupIdx = mod(idx(i)-1,numGroups)+1; % 适应度提升最大的子组
            algIdx   = ceil(idx(i)/numGroups); % 用于判断使用的优化器
            assert(algIdx==1 || algIdx==2);     
            dimIdx = allGroups{groupIdx};
            if algIdx == 1 
                [pop(:,dimIdx),fitness(:,groupIdx),xbestnew,bestvalnew,ccm(groupIdx)] = sansde(Func,func_num,dimIdx,pop(:,dimIdx),fitness(:,groupIdx),xbest,bestval,Lbound(:,dimIdx),Ubound(:,dimIdx),maxFECycle,ccm(groupIdx));               
            else
                [pop(:,dimIdx),fitness(:,groupIdx),xbestnew,bestvalnew,v(:,dimIdx)] = slpso(Func,func_num,dimIdx,pop(:,dimIdx),fitness(:,groupIdx),xbest,bestval,Lbound(:,dimIdx),Ubound(:,dimIdx),maxFECycle,v(:,dimIdx));
            end          
            if bestvalnew < bestval
                fitimp = (bestval-bestvalnew)/bestval; % 适应度提升率
                fitImpAccum(idx(i)) = (fitImpAccum(idx(i))+fitimp)/2; % 更新适应度提升矩阵  
                xbest = xbestnew;
                bestval = bestvalnew;
            else
                fitImpAccum(idx(i)) = fitImpAccum(idx(i))/2;
            end
            countFE = countFE + maxFECycle; 
            gbesthistory(countFE) = bestval;
%             fprintf(fid2, '%d, %d, %d\n', cycle, groupIdx, algIdx);
        end
        
        if(display == 1 && mod(cycle,1)==0) % 在命令行窗口输出
           fprintf(1, 'Cycle = %d, bestval = %e, component = %d, algorithm = %d, \n', cycle, bestval, groupIdx, algIdx);
%            fprintf(fid1, '%d, %e\n', countFE + groupingFE, bestval);
        end
    end
    fprintf(1, 'Cycle = %d, bestval = %e, component = %d, algorithm = %d, \n', cycle, bestval, groupIdx, algIdx);
    if countFE<FEMax
        gbesthistory(FEMax+1:MaxFEs)=gbestfitness;
    else
        if countFE>FEMax
            gbesthistory(FEMax+1:end)=[];
        end
    end
end

function [allGroups] = grouping(fun)
    filename = sprintf('./Algorithms/CCOS/CEC2010/rdg/results/F%02d', fun);
    load(filename);
    if isempty(seps)
        allGroups = nonseps;
    else
        allGroups=[{seps},nonseps];
    end
end
                
