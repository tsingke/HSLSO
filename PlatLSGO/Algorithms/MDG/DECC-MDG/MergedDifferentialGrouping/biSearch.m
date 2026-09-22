% -------
% Inputs:
% -------
%    fun        : the function suite for which the interaction structure
%                 is going to be identified in this case benchmark_func
%                 of cec'2010 or cec'2013.
%    fun_number : the function number.
%    options    : this variable contains the options such as problem
%                 dimensionality, upper and lower bounds, and 
%                 parameters base and sigma used by merged differential grouping.   
%    left_group : The current subset of the interaction to be identified in Lgroups
%    Rgroups    : The right subset group 
%    fp1        : Function value without perturbation of all variables
%    fp2        : Function value perturbing left_group.
%    fp_iR      : Function value perturbing left_group and Rgroups.
%    Rperts   : Perturbation function value of the variable set corresponding to the binary search tree on the right
%
% --------
% Outputs:
% --------
%    group   : The subset in Rgroups interacting with the current left subset left_group.
%    groupIndexs: Corresponding Index of Subset of Interaction.
%    Rperts   : Perturbation function value of the variable set corresponding to the binary search tree on the right
%    FEs      : the number of fitness evaluations used.
%
% ------

% Six evaluations are used to detect the interaction among three subsets

function [group,groupIndexs,Rperts,FEs]=biSearch(fun,fun_number,options,left_group,fp1,fp2,fp_iR,Rgroups,Rperts)
% group: the subset in the right subtree that interacts with the current subset of the left subtree
% groupIndexs: the index of the interacting subset
% Rperts: the fitness values after perturbing the variables in the right subtree of the binary tree

    base=options.base; % lower bound
    sigma=options.sigma;
    dim=options.dim; % dimensionality
    FEs=0;
    R_gnum=size(Rgroups,2); % number of subgroups in the right subtree
    Rfp=Rperts(1);          % Perturbation function value of all variables in Rgroups
    groupsQueue={};         % Save subset groups of the breadth-first traversal strategy
    dataQueue={};           % Save the function value of the subset groups of the breadth-first traversal strategy
    groupIndexsQueue={};    % Save the index position of the subset in each node in Rgroups
    pQueue={};              % Save the basic decision vector value of the current node
    
    data=[fp1,fp2,Rfp,fp_iR];
    % fp1:xl,l,l
    % fp2:xu,l,l
    % Rfp:xl,u,u
    % fp_iR:xu,u,u
    groupsQueue={Rgroups}; % Breadth-first traversal of the right subtree
    dataQueue{1}=data; % Function values of the subsets in the breadth-first traversal
    groupIndexsQueue{1}=1:R_gnum; % Indices of the corresponding positions of the subsets in the right subtree
    pQueue{1}=base * ones(1,dim); % Set all the values of the current node to the lower bound
    node_orders=1; % The node order is 1
    group={}; % The subset in the right subtree that interacts with the current subset of the left subtree
    groupIndexs=[]; % Index of the subset in the right subtree that interacts with the current subset of the left subtree
    
    while(~isempty(groupsQueue)) % This loop runs while the right subtree is not empty
        %Take out the current team leader
        cur_groups=groupsQueue{1};
        cur_data=dataQueue{1};
        cur_groupIndexs=groupIndexsQueue{1};
        cur_p=pQueue{1};
        cur_order=node_orders(1);

        groupsQueue(1)=[];
        dataQueue(1)=[];
        pQueue(1)=[];
        groupIndexsQueue(1)=[];
        node_orders(1)=[];

        delta1=cur_data(2)-cur_data(1);
        delta2=cur_data(4)-cur_data(3);

        epsilon=epsilonCalculate(cur_data(1),cur_data(2),cur_data(3),cur_data(4),dim);
%         There is an interaction between the current subset group and left_group, 
%         and the current subset group is divided into two small subset groups
        if(abs(delta1-delta2)>epsilon) % If an interaction exists
            cur_gnum=size(cur_groups,2); % Number of subgroups in the current right subtree
            if(cur_gnum==1) 
                group={group{1:end} cur_groups{1}};
                groupIndexs(end+1)=cur_groupIndexs(1);
            else % If the number of subgroups in the current right subtree is greater than 1
                median=floor(cur_gnum/2); % Split the current right subtree into two
                groups1=cur_groups(1:median);
                groupIndexs1=cur_groupIndexs(1:median);
                p_1=cur_p; % xl,l,l
                p_1(cell2mat(groups1))=base+sigma; % p_1:xl,u,l
                if(cur_order*2<=length(Rperts)&&Rperts(cur_order*2)~=0)
                    fp_1=Rperts(cur_order*2); % fp_1:xl,u,u
                else
                    fp_1=feval(fun,p_1,fun_number); % fp_1:xl,u,l
                    Rperts(cur_order*2)=fp_1; 
                    FEs=FEs+1;
                end
                p_i1=p_1;
                p_i1(left_group)=base+sigma; % p_i1:xu,u,l
                fp_i1=feval(fun,p_i1,fun_number);
                FEs=FEs+1;
                
                data1=[cur_data(1),cur_data(2),fp_1,fp_i1];
                % cur_data(1):xl,l,l
                % cur_data(2):xu,l,l
                % fp_1:xl,u,l or fp_1:xl,u,u
                % p_i1:xu,u,l
                groupsQueue{end+1}=groups1; 
                dataQueue{end+1}=data1;
                groupIndexsQueue{end+1}=groupIndexs1;
                pQueue{end+1}=cur_p;
                node_orders(end+1)=cur_order*2;

                groups2=cur_groups(median+1:cur_gnum);
                groupIndexs2=cur_groupIndexs(median+1:cur_gnum);
                data2=[fp_1,fp_i1,cur_data(3),cur_data(4)];
                % fp_1:xl,u,l or fp_1:xl,u,u
                % p_i1:xu,u,l
                % cur_data(3):xl,u,u
                % cur_data(4):xu,u,u
                groupsQueue{end+1}=groups2;
                dataQueue{end+1}=data2;
                groupIndexsQueue{end+1}=groupIndexs2;
                pQueue{end+1}=p_1; 
                node_orders(end+1)=cur_order*2+1;
            end           
        end
    end
     
end