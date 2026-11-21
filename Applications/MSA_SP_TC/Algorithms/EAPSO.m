function [gbestx,BestValue,BestCost] = EAPSO(sequence,a,lengthdata,L,historyPoints,dimension,MaxFEs)
% EAPSO MSA adaptation with an explicit total FE budget.
if nargin<7 || isempty(MaxFEs), error('EAPSO requires MaxFEs as input 7.'); end
if nargin<5 || isempty(historyPoints), historyPoints=200; end
popsize=50; nPop=popsize; nVar=dimension; VarMin=zeros(1,nVar); VarMax=ones(1,nVar);
if MaxFEs<popsize || MaxFEs~=floor(MaxFEs), error('Invalid MaxFEs.'); end
k=a;
VarSize=[1 nVar];
empty_particle.Position=[]; empty_particle.Cost=[]; empty_particle.Velocity=[];
empty_particle.Best.Position=[]; empty_particle.Best.Cost=[];
particle=repmat(empty_particle,nPop,1);
GlobalBest.Cost=-inf; GlobalBest.Position=[];
feBest=nan(MaxFEs,1); FEs=0;

for i=1:nPop
    if i==1, particle(i).Position=0.5*ones(1,nVar);
    else, particle(i).Position=VarMin+(VarMax-VarMin).*rand(1,nVar); end
    particle(i).Velocity=zeros(VarSize);
    particle(i).Cost=fitness(sequence,k,lengthdata,L,particle(i).Position);
    FEs=FEs+1;
    particle(i).Best.Position=particle(i).Position;
    particle(i).Best.Cost=particle(i).Cost;
    if particle(i).Best.Cost>GlobalBest.Cost, GlobalBest=particle(i).Best; end
    feBest(FEs)=GlobalBest.Cost;
end

cosp=zeros(1,nPop);
for i=1:nPop, cosp(i)=particle(i).Cost; end
[~,inp]=sort(cosp,'descend');
PART(1)=particle(inp(1)).Best; PART(2)=particle(inp(2)).Best;
NP=nPop; pcount=2;
GART(1)=GlobalBest; gcount=1;
VelMax=VarMax-VarMin; VelMin=-VelMax;
it=0;
BestValue=GlobalBest.Cost;
gbestx=GlobalBest.Position;

while FEs<MaxFEs
    it=it+1;
    for i=1:nPop, cosp(i)=particle(i).Best.Cost; end
    [~,ind]=sort(cosp,'descend');
    WINNER=ind(1:nPop/2); LOSER=ind(nPop/2+1:nPop);

    for i=1:nPop/2
        if FEs>=MaxFEs, break; end
        aa1=randi(length(PART)); b1=randi(length(PART));
        while aa1==b1, aa1=randi(length(PART)); b1=randi(length(PART)); end
        if PART(aa1).Cost<PART(b1).Cost, aa1=b1; end

        if length(GART)>1
            c=randi(length(GART)); d=randi(length(GART));
            while c==d, c=randi(length(GART)); d=randi(length(GART)); end
            if GART(c).Cost<GART(d).Cost, b1=d; else, b1=c; end
        else
            b1=1;
        end

        c=randi(nPop/2); q=randi(nPop/2);
        while c==q, c=randi(nPop/2); q=randi(nPop/2); end
        if particle(WINNER(q)).Best.Cost>particle(WINNER(c)).Best.Cost, c=q; end

        loser=LOSER(i);
        w=rand(1,nVar); MM=particle(loser).Position;
        F1=rand(1,nVar); F2=rand(1,nVar);

        if cosp(loser)>mean(cosp(LOSER))
            if PART(aa1).Cost>GART(b1).Cost && PART(aa1).Cost>particle(WINNER(c)).Best.Cost
                particle(loser).Velocity=w.*particle(loser).Velocity+F1.*(PART(aa1).Position-MM)+F2.*(GlobalBest.Position-MM);
            elseif PART(aa1).Cost<GART(b1).Cost && GART(b1).Cost>particle(WINNER(c)).Best.Cost
                particle(loser).Velocity=w.*particle(loser).Velocity+F1.*(GART(b1).Position-MM)+F2.*(GlobalBest.Position-MM);
            elseif PART(aa1).Cost<particle(WINNER(c)).Best.Cost && GART(b1).Cost<particle(WINNER(c)).Best.Cost
                particle(loser).Velocity=w.*particle(loser).Velocity+F1.*(particle(WINNER(c)).Best.Position-MM)+F2.*(GlobalBest.Position-MM);
            end
        else
            if PART(aa1).Cost<GART(b1).Cost && PART(aa1).Cost<particle(WINNER(c)).Best.Cost
                particle(loser).Velocity=w.*particle(loser).Velocity+F1.*(GART(b1).Position-MM)+F2.*(particle(WINNER(c)).Best.Position-particle(loser).Position);
            elseif PART(aa1).Cost>GART(b1).Cost && GART(b1).Cost<particle(WINNER(c)).Best.Cost
                particle(loser).Velocity=w.*particle(loser).Velocity+F1.*(PART(aa1).Position-MM)+F2.*(particle(WINNER(c)).Best.Position-particle(loser).Position);
            elseif PART(aa1).Cost>particle(WINNER(c)).Best.Cost && GART(b1).Cost>particle(WINNER(c)).Best.Cost
                particle(loser).Velocity=w.*particle(loser).Velocity+F1.*(PART(aa1).Position-MM)+F2.*(GART(b1).Position-particle(loser).Position);
            end
        end

        particle(loser).Velocity=max(particle(loser).Velocity,VelMin);
        particle(loser).Velocity=min(particle(loser).Velocity,VelMax);
        particle(loser).Position=particle(loser).Position+particle(loser).Velocity;
        IsOutside=(particle(loser).Position<VarMin | particle(loser).Position>VarMax);
        particle(loser).Velocity(IsOutside)=-particle(loser).Velocity(IsOutside);
        particle(loser).Position=max(particle(loser).Position,VarMin);
        particle(loser).Position=min(particle(loser).Position,VarMax);
        particle(loser).Cost=fitness(sequence,k,lengthdata,L,particle(loser).Position);
        FEs=FEs+1;

        if particle(loser).Cost>particle(loser).Best.Cost
            particle(loser).Best.Position=particle(loser).Position;
            particle(loser).Best.Cost=particle(loser).Cost;
            pcount=pcount+1;
            if pcount<=NP
                PART(pcount)=particle(loser).Best;
            else
                aa1=randi(NP); b1=randi(NP);
                while aa1==b1, aa1=randi(NP); b1=randi(NP); end
                if PART(aa1).Cost>PART(b1).Cost
                    if particle(loser).Cost>PART(b1).Cost, PART(b1)=particle(loser).Best; end
                else
                    if particle(loser).Cost>PART(aa1).Cost, PART(aa1)=particle(loser).Best; end
                end
            end
        end
        if particle(loser).Best.Cost>GlobalBest.Cost, GlobalBest=particle(loser).Best; end
        feBest(FEs)=GlobalBest.Cost;
    end

    gcount=gcount+1;
    if gcount<=NP
        GART(gcount)=GlobalBest;
    else
        aa1=randi(length(GART)); b1=randi(length(GART));
        while aa1==b1, aa1=randi(length(GART)); b1=randi(length(GART)); end
        if GART(b1).Cost<GART(aa1).Cost, aa1=b1; end
        if GlobalBest.Cost>GART(aa1).Cost, GART(aa1)=GlobalBest; end
    end

    BestValue=GlobalBest.Cost; gbestx=GlobalBest.Position;
    if mod(it,25)==0 || FEs>=MaxFEs
        fprintf('EAPSO: generation=%d, FEs=%d/%d, best SPS=%e\n',it,FEs,MaxFEs,BestValue);
    end
end

BestCost=msa_level_baseline_history(feBest,historyPoints);
gbestx=gbestx(:)';
end
