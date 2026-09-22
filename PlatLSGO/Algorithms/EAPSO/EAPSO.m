function [gbestx,BestValue,BestCost] =EAPSO(popsize,dimension,xmax,xmin,vmax,vmin,maxiter,fCalculation,FuncId)
% (fhd,nPop,nVar,VarMin,VarMax,MaxIt,X)

FEs=0;
MaxFEs=3e6;

nPop=popsize;
nVar=dimension;
VarMin=xmin.*ones(1,nVar);
VarMax=xmax.*ones(1,nVar);
MaxIt=maxiter;
fhd=fCalculation;

% MaxIt=floor(MaxIt/nPop*2);
% Use a structure array
VarSize=[1 nVar];  
empty_particle.Position=[]; % Particle position
empty_particle.Cost=[]; % Particle fitness
empty_particle.Velocity=[]; % Particle velocity
empty_particle.Best.Position=[]; % Personal best position
empty_particle.Best.Cost=[]; % Personal best fitness
particle=repmat(empty_particle,nPop,1); % nPop rows, 1 column
GlobalBest.Cost=inf;

% Swarm initialization
for i=1:nPop
    particle(i).Position=VarMin+(VarMax-VarMin).*rand(1,nVar);% Position
    particle(i).Velocity=zeros(VarSize);% Velocity
    particle(i).Cost= fhd(particle(i).Position',FuncId);% Fitness
    FEs=FEs+1;
    particle(i).Best.Position=particle(i).Position;% Personal best position
    particle(i).Best.Cost=particle(i).Cost;% Personal best fitness
    if particle(i).Best.Cost<GlobalBest.Cost
        GlobalBest=particle(i).Best;% Global best
    end
end
pcount=2;% Storage for Bt, keeping some promising historical best solutions of the particles in Xt
for i=1:nPop
    cosp(i)=particle(i).Cost; % Personal best position
end
[~,inp]=sort(cosp);% Sort by personal best position fitness
PART(1)=particle(inp(1)).Best;% Put the two best particles into At and Bt
PART(2)=particle(inp(2)).Best;
NP=nPop;
BestCost = [GlobalBest.Cost*ones((FEs),1)];
GART(1)=GlobalBest;% Ct, built to keep some promising global best solutions; its maximum length also equals the swarm size
gcount=1;% Ct
VelMax=(VarMax-VarMin);% Maximum velocity
VelMin=-VelMax;% Minimum velocity
it=2;
while FEs < MaxFEs 
    
    for i=1:nPop
        cosp(i)=particle(i).Best.Cost;% Personal best position
    end
    [~,ind]=sort(cosp);
    WINNER=ind(1:nPop/2);% Winners
    LOSER= ind(nPop/2+1:nPop);% Losers
    for i=1:nPop/2
        a=randperm(length(PART),1);
        b=randperm(length(PART),1);
        while a==b
            a=randperm(length(PART),1);
            b=randperm(length(PART),1);
        end
        if PART(a).Cost>PART(b).Cost
            a=b;
        end
        if length(GART)>1
            c=randperm(length(GART),1);
            d=randperm(length(GART),1);
            while c==d
                c=randperm(length(GART),1);
                d=randperm(length(GART),1);
            end
            if GART(c).Cost>GART(d).Cost
                b=d;
            else
                b=c;
            end
        else
            b=1;
        end
        c=randperm(nPop/2,1);
        q=randperm(nPop/2,1);
        while c==q
            c=randperm(nPop/2,1);
            q=randperm(nPop/2,1);
        end
        if  particle(WINNER(q)).Best.Cost<particle(WINNER(c)).Best.Cost
            c=q;
        end
        w=rand(1,nVar);
        MM=particle(LOSER(i)).Position;
        F1=rand(1,nVar);
        F2=rand(1,nVar);
        %% Algorithm 3
        if cosp(LOSER(i))<mean(cosp(LOSER))
            if PART(a).Cost<GART(b).Cost && PART(a).Cost<particle(WINNER(c)).Best.Cost 
                particle(LOSER(i)).Velocity=w.*particle(LOSER(i)).Velocity+F1.*(PART(a).Position-MM)+F2.*(GlobalBest.Position-MM);
            elseif PART(a).Cost>GART(b).Cost && GART(b).Cost<particle(WINNER(c)).Best.Cost
                particle(LOSER(i)).Velocity=w.*particle(LOSER(i)).Velocity+F1.*(GART(b).Position-MM)+F2.*(GlobalBest.Position-MM);
            elseif  PART(a).Cost>particle(WINNER(c)).Best.Cost &&  GART(b).Cost>particle(WINNER(c)).Best.Cost
                particle(LOSER(i)).Velocity=w.*particle(LOSER(i)).Velocity+F1.*(particle(WINNER(c)).Best.Position-MM)+F2.*(GlobalBest.Position-MM);
            end
        else
             if PART(a).Cost>GART(b).Cost && PART(a).Cost>particle(WINNER(c)).Best.Cost 
                particle(LOSER(i)).Velocity=w.*particle(LOSER(i)).Velocity+F1.*(GART(b).Position-MM)+F2.*(particle(WINNER(c)).Best.Position-particle(LOSER(i)).Position);
             elseif PART(a).Cost<GART(b).Cost && GART(b).Cost>particle(WINNER(c)).Best.Cost
                particle(LOSER(i)).Velocity=w.*particle(LOSER(i)).Velocity+F1.*(PART(a).Position-MM)+F2.*(particle(WINNER(c)).Best.Position-particle(LOSER(i)).Position);
             elseif PART(a).Cost<particle(WINNER(c)).Best.Cost && GART(b).Cost<particle(WINNER(c)).Best.Cost
                 particle(LOSER(i)).Velocity=w.*particle(LOSER(i)).Velocity+F1.*(PART(a).Position-MM)+F2.*(GART(b).Position-particle(LOSER(i)).Position); 
             end
        end
        particle(LOSER(i)).Velocity = max(particle(LOSER(i)).Velocity,VelMin);% Velocity boundary control
        particle(LOSER(i)).Velocity = min(particle(LOSER(i)).Velocity,VelMax);
        particle(LOSER(i)).Position = particle(LOSER(i)).Position + particle(LOSER(i)).Velocity;
        IsOutside=(particle(LOSER(i)).Position<VarMin | particle(LOSER(i)).Position>VarMax);% Position boundary control
        particle(LOSER(i)).Velocity(IsOutside)=-particle(LOSER(i)).Velocity(IsOutside);
        particle(LOSER(i)).Position = max(particle(LOSER(i)).Position,VarMin);
        particle(LOSER(i)).Position = min(particle(LOSER(i)).Position,VarMax);
        particle(LOSER(i)).Cost = fhd(particle(LOSER(i)).Position',FuncId);
        FEs=FEs+1;
        % Update Bt
        if particle(LOSER(i)).Cost<particle(LOSER(i)).Best.Cost
            particle(LOSER(i)).Best.Position=particle(LOSER(i)).Position;
            particle(LOSER(i)).Best.Cost=particle(LOSER(i)).Cost;
            pcount=pcount+1;
            if pcount<=NP
                PART(pcount).Position=particle(LOSER(i)).Position;
                PART(pcount).Cost=particle(LOSER(i)).Cost;
            else
                a=randperm(NP,1);
                b=randperm(NP,1);
                while a==b
                    a=randperm(NP,1);
                    b=randperm(NP,1);
                end
                if PART(a).Cost<PART(b).Cost
                    if particle(LOSER(i)).Cost<PART(b).Cost
                        PART(b).Cost=particle(LOSER(i)).Cost;
                        PART(b).Position=particle(LOSER(i)).Position;
                    end
                else
                    if particle(LOSER(i)).Cost<PART(a).Cost
                        PART(a).Cost=particle(LOSER(i)).Cost;
                        PART(a).Position=particle(LOSER(i)).Position;
                    end
                end
            end
        end
        if particle(LOSER(i)).Best.Cost<GlobalBest.Cost
            GlobalBest=particle(LOSER(i)).Best;
        end
    end
    % Update Ct
    gcount=gcount+1;
    if gcount<=NP
        GART(gcount).Cost= GlobalBest.Cost;
        GART(gcount).Position= GlobalBest.Position;
    else
        a=randperm(length(GART),1);
        b=randperm(length(GART),1);
        while a==b
            a=randperm(length(GART),1);
            b=randperm(length(GART),1);
        end
        if GART(b).Cost>GART(a).Cost
            a=b;
        end
        if  GlobalBest.Cost<GART(a).Cost
            GART(a).Cost= GlobalBest.Cost;
            GART(a).Position= GlobalBest.Position;
        end   
    end
    BestValue=GlobalBest.Cost;
    gbestx=GlobalBest.Position;

    FES0 = length(BestCost);
    BestCost = [BestCost; BestValue*ones((FEs-FES0),1)];

    if mod(FEs, floor(MaxFEs/10)) == 0 && FEs <= MaxFEs
        fprintf("EAPSO  FE %d  best = %e\n",FEs,BestValue);
    end
    
    if FEs >= MaxFEs
        break;
    end
    it = it+1;
end
if FEs<MaxFEs
    BestCost(FEs+1:MaxFEs)=BestValue;
else
    if FEs>MaxFEs
        BestCost(MaxFEs+1:end)=[];
    end
end
end


