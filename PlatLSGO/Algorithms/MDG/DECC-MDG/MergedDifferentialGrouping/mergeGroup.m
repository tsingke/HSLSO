% -------
% Inputs:
% -------
%    fun        : the function suite for which the interaction structure
%                 is going to be identified in this case benchmark_func
%                 of cec'2010 or cec'2013.
%
%    fun_number : the function number.
%
%    options    : this variable contains the options such as problem
%                 dimensionality, upper and lower bounds, and 
%                 parameters base and sigma used by merged differential grouping.
%    
%    dims       : Variable vector currently to be grouped
%
%    fp1        : Function value without perturbation of all variables
%
%    perturbed_values   : Function value after perturbing each variable
%
%
% --------
% Outputs:
% --------
%    groups   : The grouping result of the current vector dims.
%    groups_perturb: Function value after perturbing dims.
%    FEs      : the number of fitness evaluations used.
%
% ------
function [groups,groups_perturb,FEs]=mergeGroup(fun,fun_number,options,dims,fp1,perturbed_values)
    % dims stores all the non-separable variables
    % fp1 is the fitness value with all the variables at their lower bounds
    % The fitness values with a perturbation added at the i-th dimension are stored in perturbed_values, so there is no need to recompute them
    dim=options.dim;% Here dim is the overall dimensionality
    base=options.base;
    sigma=options.sigma;
    FEs=0;
    dim_len=length(dims);% Compute the number of non-separable variables
    groups={};% Used to record all the sets of non-separable variables
    
    %% If there is only one or two non-separable variables
    if(dim_len==1 || dim_len==2) 
        %A single non-separable variable, return directly
        if(dim_len==1)
            groups_perturb=perturbed_values(dims);% The function value of this group after the perturbation is added
            groups{1}=dims;
        else% Two non-separable variables: determine whether they interact. With only two non-separable variables, can they still be non-interacting?
            p=base * ones(1,dim);% Set to the lower bound
            p(dims)=base+sigma;% Set to the upper bound
            fp=feval(fun,p,fun_number);% Compute the fitness
            FEs=FEs+1;
            
            delta1=perturbed_values(dims(1))-fp1;
            delta2=fp-perturbed_values(dims(2));
            epsilon=epsilonCalculate(fp1,perturbed_values(dims(1)),perturbed_values(dims(2)),fp,dim);
            
            groups_perturb=fp;% The value after adding the perturbation to both decision variables contained in the non-separable variable group
            if (abs(delta1-delta2)<epsilon) % No interaction, so it is divided into two groups
                groups{1}=dims(1);
                groups{2}=dims(2);
            else
                groups{1}=dims';
            end
        end
        return;
    end
    
    %% The variable set dims is divided into two subsets
    median=floor(dim_len/2); % Compute the middle value
    Ldims=dims(1:median);% Divide dims into two equally sized and mutually exclusive subsets
    Rdims=dims(median+1:dim_len);
    
    %Two subsets are grouped separately, recursive call
    [Lgroups,Lgroups_perturb,LFEs]=mergeGroup(fun,fun_number,options,Ldims,fp1,perturbed_values);
    [Rgroups,Rgroups_perturb,RFEs]=mergeGroup(fun,fun_number,options,Rdims,fp1,perturbed_values);
    
    FEs=LFEs+RFEs;
    L_gnum=size(Lgroups,2); % Record how many non-separable variable groups are contained
    R_gnum=size(Rgroups,2);
    
    %% Merge non-separable subsets between two subset groups
    %Determine whether there is an interaction between two subset groups
    Lfp=Lgroups_perturb; % Record the fitness value after the group perturbation
    Rfp=Rgroups_perturb;
    
    p=base * ones(1,dim);
    p(dims)=base+sigma;% Add the perturbation to all the non-separable variables
    fp=feval(fun,p,fun_number);
    FEs=FEs+1;
    
    delta1=Lfp-fp1;
    delta2=fp-Rfp;
    epsilon=epsilonCalculate(fp1,Lfp,Rfp,fp,dim);
    groups_perturb=fp;
    
    
    % If there is an interaction between the two subtrees
    if(abs(delta1-delta2)>epsilon) 
        if(L_gnum==1&&R_gnum==1) % If both the left and the right subtree contain only one variable, merge them directly
            Rgroups{1}=[Lgroups{1} Rgroups{1}];
            groups=Rgroups;
            return ; 
        end
        
        %First find out the subset of the left subset group that interacts with the right subset group
        % If there is more than one variable in the left and right subtrees, continue the test
        [Lgroup,LgroupIndexs,~,FE]=biSearch(fun,fun_number,options,Rdims',fp1,Rfp,fp,Lgroups,Lfp);
        FEs=FEs+FE;
        if(~isempty(LgroupIndexs))
            lnum=length(LgroupIndexs);
            %There is only a subset on the right
            if(R_gnum==1)
                Rgroups{1}=[cell2mat(Lgroup) Rgroups{1}];
                LgroupIndexs=sort(LgroupIndexs,'descend');
                for i=LgroupIndexs
                    Lgroups(i)=[];
                end
                groups=[Lgroups Rgroups];
                return ; 
            end
            Rinteract_Indexs=cell(1,lnum);
            Rperts=Rfp;%Perturbation function value of the variable set corresponding to the binary search tree on the right
            for i=1:lnum
                if(L_gnum==1)
                    fp_i=Lfp;
                    fp_iR=fp;
                else
                    p_i=base * ones(1,dim);
                    p_i(Lgroup{i})=base+sigma;
                    fp_i=feval(fun,p_i,fun_number);
                    p_iR=p_i; 
                    iR=[Lgroup{i}, Rdims'];
                    p_iR(iR)=base+sigma;
                    fp_iR=feval(fun,p_iR,fun_number);
                    FEs=FEs+2;
                end            
                [~,RgroupIndexs,Rperts,FE]=biSearch(fun,fun_number,options,Lgroup{i},fp1,fp_i,fp_iR,Rgroups,Rperts);
                FEs=FEs+FE;
                Rinteract_Indexs{i}=RgroupIndexs;
            end
            
            %Merge interaction group
            [merged_groups]=mergeInteractionGroup(LgroupIndexs,Rinteract_Indexs,Lgroups,Rgroups);
            LgroupIndexs=sort(LgroupIndexs,'descend');
            for j=LgroupIndexs
                Lgroups(j)=[];
            end
            list_r=[];
            for k=1:lnum
                list_r=union(list_r,Rinteract_Indexs{k});
            end
            list_r=sort(list_r,'descend');
            for j=list_r
                Rgroups(j)=[];
            end
            Rgroups=[Rgroups merged_groups];
        end
    end
        
    groups=[Lgroups Rgroups];    
           
end