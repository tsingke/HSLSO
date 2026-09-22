% Author: Dr. Zhyenu Yang
% Modified by: Mohammad Nabi Omidvar
% email address: mn.omidvar AT gmail.com
%
% ------------
% Description:
% ------------
% This file is an implementation of cooperative co-evolution which
% uses SaNSDE algorithm as subcomponent optimizer.
%
% -----------
% References:
% -----------
% Omidvar, M.N.; Li, X.; Mei, Y.; Yao, X., "Cooperative Co-evolution with
% Differential Grouping for Large Scale Optimization," Evolutionary Computation,
% IEEE Transactions on, vol.PP, no.99, pp.1,1, 0
% http://dx.doi.org/10.1109/TEVC.2013.2281543
%
% --------
% License:
% --------
% This program is to be used under the terms of the GNU General Public License 
% (http://www.gnu.org/copyleft/gpl.html).
% Author: Mohammad Nabi Omidvar
% e-mail: mn.omidvar AT gmail.com
% Copyright notice: (c) 2013 Mohammad Nabi Omidvar


function [best,bestval,gbesthistory] = DECC_MDG(popsize, dim, xmax, xmin, ~, ~, itermax, fname, func_num)

    % for fitness trace
    tracerst = [];

    problem = 2013;

    MaxFEs = 3e6;

    decResults = sprintf('./Algorithms/MDG/DECC-MDG/MergedDifferentialGrouping/results2013/F%02d',func_num);
    load (decResults);
%     FEs = Max_FEs - FEs;

    Lbound = xmin.*ones(popsize,dim);
    Ubound = xmax.*ones(popsize,dim);

    % the initial population
    pop = Lbound + rand(popsize, dim) .* (Ubound-Lbound);
    for i = 1:popsize
        val(i) = fname(pop(i,:)', func_num);
    end
    [bestval, ibest] = min(val);%Return the minimum value of each column and the row number corresponding to the value (starting from 1)
    bestmem = pop(ibest, :);%Get the best fit individual


    % the initial crossover rate for SaNSDE
    group = {};
    ccm = 0.5;%Mean of the Gaussian random function for the CRm crossover rate; updated every 25 generations
    sansde_iter = 1;
    Cycle = 0;
    iter = 0;
    delta = 0;

    FE = FEs+popsize;
    gbesthistory = [bestval*ones((FE),1)];
    
    group = diff_grouping(func_num,problem);%Returns the grouping result of differential grouping
    group_num = size(group, 2);%Returns the number of groups
        
    
    display = 1;
    frequency = 100;

    while (FE < MaxFEs)

        for i = 1:group_num
            oneitermax = sansde_iter;
            if (iter + oneitermax >= itermax)
                oneitermax = itermax - iter;
            end
            if (oneitermax == 0)
                break;
            end

            dim_index = group{i};%Returns the dimension (decision variable) indices of the i-th subcomponent
            subpop = pop(:, dim_index); 
            subLbound = Lbound(:, dim_index);        
            subUbound = Ubound(:, dim_index);

                [subpopnew, bestmemnew, bestvalnew, tracerst, ccm] = sansde(fname, func_num, dim_index, subpop, bestmem, bestval, subLbound, subUbound, oneitermax, ccm,Cycle);

                FE = FE + popsize;

                iter = iter + oneitermax;
                
                %The output format is real numbers in scientific notation
%                 fprintf(fid, '%e\n', tracerst); 

                
                pop(:, dim_index) = subpopnew;
                bestmem = bestmemnew;
                bestval = bestvalnew;

                nonsep = [];
                sep = [];

                if(display == 1)
                    fprintf(1, 'Cycle = %d, bestval = %e, Group = %d *\n',  Cycle, bestval, i);
                end

                if(iter > itermax)
                    break;
                end


        end

     
        val = fname(pop', func_num);
        [best, ibest] = min(val);
        if (best < bestval)
            bestval = best;
            bestmem = pop(ibest, :);
        end
        FES0 = length(gbesthistory);
        gbesthistory = [gbesthistory; bestval*ones((FE-FES0),1)];

        Cycle = Cycle + 1;
        
        
    end
    if FE<MaxFEs
        gbesthistory(MaxFEs+1:MaxFEs)=bestval;
    else
        if FE>MaxFEs
            gbesthistory(MaxFEs+1:end)=[];
        end
    end
end

function group = diff_grouping(fun,problem)   
    if(problem==2010)
        filename=sprintf('./Algorithms/MDG/DECC-MDG/MergedDifferentialGrouping/results2010/F%02d',fun);
    else
        filename=sprintf('./Algorithms/MDG/DECC-MDG/MergedDifferentialGrouping/results2013/F%02d',fun);
    end
    load(filename);
    group = nonseps;
    if(~isempty(seps))
        group = {group{1:end} seps};
    end
end


% Optimize the subcomponent using SaNSDE
% The SaNSDE algorithm can be found in:
% Zhenyu Yang, Ke Tang and Xin Yao, "Self-adaptive Differential Evolution with
% Neightborhood Search", in Proceedings of the 2008 IEEE Congress on 
% Evolutionary Computation (CEC2008), Hongkong, China, 2008, pp. 1110-1116.

function [popnew, bestmemnew, bestvalnew, tracerst, ccm] = sansde(fname, func_num, dim_index, pop, bestmem, bestval, Lbound, Ubound, itermax, ccm,cycle);

    [popsize, dim] = size(pop);
    NP = popsize;
    D = dim;
    tracerst = [];

    F = zeros(NP,1);
    %cc=zeros(NP*3, 1);

    linkp = 0.5;
    l1 = 1;l2 = 1;nl1 = 1;nl2 = 1;

    fp = 0.5;
    ns1 = 1; nf1 = 1; ns2 = 1; nf2 = 1;

    pm1 = zeros(NP,D);              % initialize population matrix 1
    pm2 = zeros(NP,D);              % initialize population matrix 2
    pm3 = zeros(NP,D);              % initialize population matrix 3
    pm4 = zeros(NP,D);              % initialize population matrix 4
    pm5 = zeros(NP,D);              % initialize population matrix 5
    bm  = zeros(NP,D);              % initialize DE_gbestber  matrix
    ui  = zeros(NP,D);              % intermediate population of perturbed vectors
    mui = zeros(NP,D);              % mask for intermediate population
    mpo = zeros(NP,D);              % mask for old population
    rot = (0:1:NP-1);               % rotating index array (size NP)
    rotd= (0:1:D-1);                % rotating index array (size D)
    rt  = zeros(NP);                % another rotating index array
    rtd = zeros(D);                 % rotating index array for exponential crossover
    a1  = zeros(NP);                % index array
    a2  = zeros(NP);                % index array
    a3  = zeros(NP);                % index array
    a4  = zeros(NP);                % index array
    a5  = zeros(NP);                % index array
    ind = zeros(4);

            cc_rec = [];
            f_rec = [];
    %Replace the individuals of the variables other than the subcomponent to be improved with the representative elements of the best individual
    gpop = ones(popsize, 1) * bestmem;
    gpop(:, dim_index) = pop;
    %Update best
    val = fname(gpop', func_num);
    [best, ibest] = min(val);
    subbestmem = pop(ibest, :);

    if (best < bestval)
        bestval = best;
        bestmem = gpop(ibest, :);
    end

    iter = 0;
    %iter_c=cycle;
    while iter < itermax
        popold = pop;                   % save the old population
        
        ind = randperm(4);              % index pointer array, returns a random permutation of the integers from 1 to 4
        
        a1  = randperm(NP);             % shuffle locations of vectors
        rt = rem(rot+ind(1),NP);        % rotate indices by ind(1) positions
        a2  = a1(rt+1);                 % rotate vector locations
        rt = rem(rot+ind(2),NP);        %rem takes the remainder
        a3  = a2(rt+1);                
        rt = rem(rot+ind(3),NP);
        a4  = a3(rt+1);               
        rt = rem(rot+ind(4),NP);
        a5  = a4(rt+1); 
        
        pm1 = popold(a1,:);             % shuffled population 1
        pm2 = popold(a2,:);             % shuffled population 2
        pm3 = popold(a3,:);             % shuffled population 3
        pm4 = popold(a4,:);             % shuffled population 4
        pm5 = popold(a5,:);             % shuffled population 5
        
        bm = ones(NP, 1) * subbestmem;
        %Update CRm (ccm) every 25 generations
        if rem(iter,24)==0
            if (iter~=0) && (~isempty(cc_rec))
                ccm = sum(f_rec.*cc_rec)/sum(f_rec);%CRm
            end
            cc_rec = [];
            f_rec = [];
        end
        %Update the crossover rate cc every 5 generations; an NP-dimensional matrix

        if rem(iter,5)==0
            cc = normrnd(ccm, 0.1, NP*3, 1);%Generates a normally distributed random matrix with (NP*3) rows and 1 column
            index = find((cc < 1) & (cc > 0));
            cc = cc(index(1:NP));
        end
        %Adaptive update of the scaling factor F
        fst1 = (rand(NP,1) <= fp);
        fst2 = 1-fst1;

        fst1_index = find(fst1 ~= 0);
        fst2_index = find(fst1 == 0);

        tmp = normrnd(0.5, 0.3, NP, 1);
        F(fst1_index) = tmp(fst1_index);

        tmp = normrnd(0, 1, NP, 1) ./ normrnd(0, 1, NP, 1);
        F(fst2_index) = tmp(fst2_index);

        F = abs(F);
        
        % all random numbers < CR are 1, 0 otherwise
        aa = rand(NP,D) < repmat(cc,1,D);%repmat produces a matrix of size [size(cc,1)*1, size(cc,2)*D], i.e. [NP,D], in which every element of a row equals cc[i]
        index = find(sum(aa') == 0);%aa' is the transpose; sum gives the sum of each column, returned as a row vector
        tmpsize = size(index, 2);
        for k=1:tmpsize
            bb = ceil(D*rand);%Round up
            aa(index(k), bb) = 1;%Ensure that every individual undergoes crossover so that ui is not a duplicate of xi
        end
            
        mui=aa;
        mpo = mui < 0.5;                % inverse mask to mui

        aaa = (rand(NP,1) <= linkp);
        aindex=find(aaa == 0);
        bindex=find(aaa ~= 0);
        
        if ~isempty(bindex)
            % mutation
            ui(bindex,:) = popold(bindex,:)+repmat(F(bindex,:),1,D).*(bm(bindex,:)-popold(bindex,:)) + repmat(F(bindex,:),1,D).*(pm1(bindex,:) - pm2(bindex,:) + pm3(bindex,:) - pm4(bindex,:));
            % crossover
            ui(bindex,:) = popold(bindex,:).*mpo(bindex,:) + ui(bindex,:).*mui(bindex,:);
        end
        if ~isempty(aindex)
            ui(aindex,:) = pm3(aindex,:) + repmat(F(aindex,:),1,D).*(pm1(aindex,:) - pm2(aindex,:));
            ui(aindex,:) = popold(aindex,:).*mpo(aindex,:) + ui(aindex,:).*mui(aindex,:);
        end
        bbb=1-aaa; 

        %-----Select which vectors are allowed to enter the new population-------
        index = find(ui > Ubound);
        ui(index) = Ubound(index) - mod((ui(index)-Ubound(index)), (Ubound(index)-Lbound(index)));
        index = find(ui < Lbound);
        ui(index) = Lbound(index) + mod((Lbound(index)-ui(index)), (Ubound(index)-Lbound(index)));
        
        gpop(:, dim_index) = ui;
        tempval = fname(gpop', func_num);
        
        for i=1:NP
            if (tempval(i) <= val(i))
                if (tempval(i) < val(i))
                    cc_rec = [cc_rec cc(i,1)];
                    f_rec = [f_rec (val(i) - tempval(i))];
                end

                pop(i,:) = ui(i,:);  
                val(i)   = tempval(i);  
                
                l1 = l1 + aaa(i);
                l2 = l2 + bbb(i);

                ns1 = ns1 + fst1(i);
                ns2 = ns2 + fst2(i);
            else
                nl1 = nl1 + aaa(i);
                nl2 = nl2 + bbb(i); 

                nf1 = nf1 + fst1(i);
                nf2 = nf2 + fst2(i);
            end
        end 
        
        if (rem(iter,24) == 0) && (iter~=0)
            linkp = (l1/(l1+nl1))/(l1/(l1+nl1)+l2/(l2+nl2));
            l1 = 1;l2 = 1; nl1 = 1; nl2 = 1;
            fp = (ns1 * (ns2 + nf2))/(ns2 * (ns1 + nf1) + ns1 * (ns2 + nf2));
            ns1 = 1; nf1 = 1; ns2 = 1; nf2 = 1;
        end    
        
        [best, ibest] = min(val);
        subbestmem = pop(ibest, :);
        
        if (best < bestval)
           bestval = best;
           bestmem(dim_index) = pop(ibest, :);
        end
       
        tracerst = [tracerst; bestval];
        iter = iter + 1;
    end

    popnew = pop;
    bestmemnew = bestmem;
    bestvalnew = bestval;

end

