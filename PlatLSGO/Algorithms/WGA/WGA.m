function [gbestx,gbestfitness,gbesthistory]=WGA(popsize,dimension,xmax,xmin,vmax,vmin,maxiter,Func,FuncId)
% M. Ghasemi, A. Rahimnejad, R. Hemmati, E. Akbari, and S. A. Gadsden,
% “Wild Geese Algorithm: A novel algorithm for large scale optimization
% based on the natural life and death of wild geese,” 
% Array, vol. 11, p. 100074, Sep. 2021, 
% doi: 10.1016/J.ARRAY.2021.100074.
% WebLink to Full Text Article: https://doi.org/10.1016/J.ARRAY.2021.100074

% nPop = popsize;
nPop=popsize;
nVar = dimension;
% nVar=1000;
nPop_Initial=nPop; % Initial swarm size
nPop_Final=popsize/4; % Final swarm size
CostFunction = Func;
MaxNFE=10000*dimension;

% Approximate maximum number of iterations
MaxIter0=ceil(MaxNFE/((nPop_Initial+nPop_Final)/2));   % Approximate Maximum Iterations

Cr=0.5;

% xmin=-600;
% xmax=-xmin;

%%%%%%%%%%%

NFE=0;
Gbest.Position=[];
Gbest.Cost=inf;
% BestCosts=nan(1,MaxIter0);
% nfe=BestCosts;

for i=1:nPop
    Velocity(i,:)=zeros(1,nVar); %#ok<*SAGROW>
    Position(i,:)=xmin+(xmax-xmin)*rand(1,nVar);
    Cost(i)=CostFunction(Position(i,:)',FuncId);
    NFE=NFE+1;
    PbestPosition(i,:)=Position(i,:);% Personal best
    PbestCost(i)=Cost(i);
    PbestVel(i,:)=Velocity(i,:);
    
    if PbestCost(i)<Gbest.Cost
        Gbest.Position=PbestPosition(i,:);
        Gbest.Cost=PbestCost(i);
        Gbest.Velocity= PbestVel(i,:);
    end

    gbesthistory(NFE)=Gbest.Cost;
    if mod(NFE, floor(MaxNFE/10)) == 0 && NFE <= MaxNFE
        fprintf("WGA  FE %d  best = %e\n",NFE,Gbest.Cost);
    end
end
% NFE=NFE+nPop;

iter=0;
while NFE<=MaxNFE
    iter=iter+1;
    
    [hh, gg]=sort(PbestCost); % Sort in ascending order
    
    nPop=(nPop_Initial-1)-((nPop_Initial-nPop_Final)*(NFE/MaxNFE)); % Swarm size shrinks linearly
    nPop=round(nPop+1);
    nPop=max(nPop,nPop_Final);
    nPop=min(nPop_Initial,nPop);
    B6=nPop_Initial-nPop_Final;
    
    for eee=1:nPop
        
        if B6==0 % If the initial swarm size == the final swarm size, this branch should never be reached
            i=eee;
        else
            i=gg(eee);% Otherwise come here; i is the index of the individual whose sorted fitness rank is eee
        end
        [~, f2]=find(gg==i); % f2 is that individual
        
        %%% Worst
        if f2==nPop % Global worst
            f2=0;
        end
        jj1=gg(1,f2+1); % Index of the individual one rank behind

        %%% BETTER
        [~, f2]=find(gg==i);
        tt=1;
        if f2==1 % Global best
            f2=nPop+1;
            tt=-1;
        end
        
        jj2=gg(1,f2-1); % Index of the individual one rank ahead
        if f2==2
            f2=nPop+2;
        end
        
        jj3=gg(1,f2-2); % Index of the individual two ranks ahead
        jjj=gg(1,1); % Global best
        ff1=gg(1,end); % Global worst
        
        Velocity(i,:)= (rand(1,nVar).*Velocity(i,:)+rand(1,nVar).*(Velocity(jj2,:)-Velocity(jj1,:)))+rand(1,nVar).*(PbestPosition(i,:)-Position(jj1,:))+rand(1,nVar).*(PbestPosition(jj2,:)-Position(i,:))-rand(1,nVar).*(PbestPosition(jj1,:)-Position(jj3,:))+rand(1,nVar).*(PbestPosition(jj3,:)-Position(jj2,:));%%ORIGINAL
        
        BB=(PbestCost(jj2))/(PbestCost(i));
        GG=(PbestCost(jjj))/(PbestCost(i));
        
        Position(i,:)=PbestPosition(i,:)+rand(1,nVar).*rand(1,nVar).*(( PbestPosition(jj2,:)+Gbest.Position-2*PbestPosition(i,:))+(Velocity(i,:)));
        
        f1=(Gbest.Cost)/(PbestCost(i)+Gbest.Cost);
        f0=(PbestCost(jj2))/(PbestCost(jj2)+PbestCost(i));
        
        DE1=((PbestPosition(jj2,:)-PbestPosition(i,:)));
        
        % Mutation
        for ww=1:nVar
            if rand<Cr
                Position(i,ww)=PbestPosition(i,ww)+rand*rand*(DE1(ww));
            end
        end
        
        Position(i,:)=min(max(Position(i,:),xmin),xmax); % Boundary control
        
        Cost(i)=CostFunction(Position(i,:)',FuncId);
        NFE=NFE+1;
        
        if Cost(i)<PbestCost(i)
            PbestPosition(i,:)=Position(i,:);
            PbestCost(i)=Cost(i);
            PbestVel(i,:)=Velocity(i,:);
            
            if PbestCost(i)<Gbest.Cost
                Gbest.Position=PbestPosition(i,:);
                Gbest.Cost=PbestCost(i);
                Gbest.Velocity= PbestVel(i,:);
            end
        end
        gbesthistory(NFE)=Gbest.Cost;
        if mod(NFE, floor(MaxNFE/10)) == 0 && NFE <= MaxNFE
            fprintf("WGA  FE %d  best = %e\n",NFE,Gbest.Cost);
        end
        gbestx = Gbest.Position;
        gbestfitness = Gbest.Cost;
        if NFE>=MaxNFE
            break;
        end
    end

    if NFE<MaxNFE
        gbesthistory(NFE+1:MaxNFE)=gbestfitness;
    else
        if NFE>MaxNFE
            gbesthistory(MaxNFE+1:end)=[];
        end
    end
end

    
    
    % NFE=NFE+nPop;
    % nfe(iter)=NFE;
    % BestCosts(iter)=Gbest.Cost;
    
%     disp(['NFE ' num2str(NFE) ':   Best Cost = ' num2str(Gbest.Cost)]);

end
% BestCosts(iter+1:end)=[];
% 
% plot(nfe,BestCosts, 'LineWidth',2)
% xlabel('NFE')
% ylabel('Objective Value')
