% Author: Dr. Yuan SUN
% email address: yuan.sun@unimelb.edu.au OR suiyuanpku@gmail.com
%
% ------------
% Description:
% ------------
% ccos - This function implements the CC framework that can select an appropriate 
%        optimizer from a given algorith pool for solving large-scale problems.
function [xbest,bestval,gbesthistory] = CCOS(NP,dim,xmax,xmin,~,~,~,Func,func_num)
%     global gbesthistory;
%     global countFE;

    decResults = sprintf('./Algorithms/CCOS/CEC2013/rdg/results/F%02d', func_num);
    load (decResults);
    FEMax = 3e6; % The number of evaluations used in the optimization phase equals the maximum number of evaluations minus those used in the grouping phase

    Lbound = xmin.*ones(NP,dim);
    Ubound = xmax.*ones(NP,dim);

    % trace for the fitness value
    groupingFE = 3e6 - FEs; % FEs used in decomposition
    
    countFE = FEs; % Number of evaluations (FEs)
    maxFECycle = 10000; % Maximum number of evaluations used per cycle
  
    % grouping cell
    allGroups = grouping(func_num); % Load the grouping result of the RDG method
    numGroups = size(allGroups, 2); % Number of groups
    
    % initialization for sansde
    popsize = NP; % Swarm size; here NP=100, defined in the run function
    ccm = 0.5*ones(1,numGroups);
    pop = Lbound + rand(popsize, dim) .* (Ubound-Lbound); % Swarm initialization
    for i = 1:popsize
        val(i) = Func(pop(i,:)',func_num);
    end
    [bestval, ibest] = min(val); % Get the best fitness value and the individual with the best fitness value
    xbest = pop(ibest, :);
    countFE=countFE+popsize; % Number of evaluations (FEs)
%     fprintf(fid1, '%d, %e\n', groupingFE, bestval);
    
    % Initialization for slpso
    v = zeros(NP, dim);
    
    % Initialization of fitness improvement array
    fitImpAccum = zeros(1,2*numGroups);
    fitness = repmat(val', [1 numGroups]);
     
    cycle = 0;
    display = 1; % 1 display results; 0 not display results 
    while (countFE < FEMax)
        cycle = cycle + 1;   
        maxVal = max(fitImpAccum); % Get the maximum value of the fitness improvement array
        idx    = find(fitImpAccum == maxVal); % Get the index of the maximum value of the fitness improvement array
        for i = 1:length(idx)    
            groupIdx = mod(idx(i)-1,numGroups)+1; % Subgroup with the largest fitness improvement
            algIdx   = ceil(idx(i)/numGroups); % Used to determine which optimizer is used
            assert(algIdx==1 || algIdx==2);     
            dimIdx = allGroups{groupIdx};
            if algIdx == 1 
                [pop(:,dimIdx),fitness(:,groupIdx),xbestnew,bestvalnew,ccm(groupIdx)] = sansde(Func,func_num,dimIdx,pop(:,dimIdx),fitness(:,groupIdx),xbest,bestval,Lbound(:,dimIdx),Ubound(:,dimIdx),maxFECycle,ccm(groupIdx));               
            else
                [pop(:,dimIdx),fitness(:,groupIdx),xbestnew,bestvalnew,v(:,dimIdx)] = slpso(Func,func_num,dimIdx,pop(:,dimIdx),fitness(:,groupIdx),xbest,bestval,Lbound(:,dimIdx),Ubound(:,dimIdx),maxFECycle,v(:,dimIdx));
            end          
            if bestvalnew < bestval
                fitimp = (bestval-bestvalnew)/bestval; % Fitness improvement rate
                fitImpAccum(idx(i)) = (fitImpAccum(idx(i))+fitimp)/2; % Update the fitness improvement array
                xbest = xbestnew;
                bestval = bestvalnew;
            else
                fitImpAccum(idx(i)) = fitImpAccum(idx(i))/2;
            end
            countFE = countFE + maxFECycle; 
            gbesthistory(countFE) = bestval;
%             fprintf(fid2, '%d, %d, %d\n', cycle, groupIdx, algIdx);
        end
        
        if(display == 1 && mod(cycle,1)==0) % Output in the command window
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
                
