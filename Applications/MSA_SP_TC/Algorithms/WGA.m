function [gbestx,gbestfitness,gbesthistory] = WGA(sequence,a,lengthdata,L,historyPoints,dimension,MaxFEs)
% Wild Geese Algorithm, MSA adaptation with a fixed total FE budget.
% Population decreases from 120 to 30 as the FE budget is consumed.
if nargin<7 || isempty(MaxFEs), error('WGA requires MaxFEs as input 7.'); end
if nargin<5 || isempty(historyPoints), historyPoints=200; end
nPopInitial=120; nPopFinal=30; nVar=dimension; xmin=0; xmax=1; Cr=0.5;
if MaxFEs<nPopInitial || MaxFEs~=floor(MaxFEs), error('MaxFEs must be >= 120.'); end

Position=zeros(nPopInitial,nVar); Velocity=zeros(nPopInitial,nVar);
PbestPosition=zeros(nPopInitial,nVar); PbestVel=zeros(nPopInitial,nVar);
Cost=zeros(nPopInitial,1); PbestCost=zeros(nPopInitial,1);
feBest=nan(MaxFEs,1); FEs=0; Gbest.Position=[]; Gbest.Cost=-inf; Gbest.Velocity=[];

for i=1:nPopInitial
    Velocity(i,:)=zeros(1,nVar);
    if i==1, Position(i,:)=0.5*ones(1,nVar);
    else, Position(i,:)=xmin+(xmax-xmin)*rand(1,nVar); end
    Cost(i)=fitness(sequence,a,lengthdata,L,Position(i,:));
    FEs=FEs+1;
    PbestPosition(i,:)=Position(i,:); PbestCost(i)=Cost(i); PbestVel(i,:)=Velocity(i,:);
    if PbestCost(i)>Gbest.Cost
        Gbest.Position=PbestPosition(i,:); Gbest.Cost=PbestCost(i); Gbest.Velocity=PbestVel(i,:);
    end
    feBest(FEs)=Gbest.Cost;
end

iter=0;
while FEs<MaxFEs
    iter=iter+1;
    [~,gg]=sort(PbestCost,'descend');

    % Shrink the active population over the consumed FE budget.
    progress=min(FEs/MaxFEs,1);
    nPop=round(nPopInitial-(nPopInitial-nPopFinal)*progress);
    nPop=max(nPopFinal,min(nPopInitial,nPop));

    for eee=1:nPop
        if FEs>=MaxFEs, break; end
        i=gg(eee);
        f2=find(gg(1:nPop)==i,1,'first');
        if isempty(f2), continue; end

        % Preserve the neighboring-rank construction of the source implementation.
        if f2==nPop, f2=0; end
        jj1=gg(f2+1);
        f2=find(gg(1:nPop)==i,1,'first');
        if f2==1, f2=nPop+1; end
        jj2=gg(f2-1);
        if f2==2, f2=nPop+2; end
        jj3=gg(f2-2);
        jjj=gg(1);

        Velocity(i,:)=(rand(1,nVar).*Velocity(i,:) ...
            +rand(1,nVar).*(Velocity(jj2,:)-Velocity(jj1,:))) ...
            +rand(1,nVar).*(PbestPosition(i,:)-Position(jj1,:)) ...
            +rand(1,nVar).*(PbestPosition(jj2,:)-Position(i,:)) ...
            -rand(1,nVar).*(PbestPosition(jj1,:)-PbestPosition(jj3,:)) ...
            +rand(1,nVar).*(PbestPosition(jj3,:)-PbestPosition(jj2,:));

        Position(i,:)=PbestPosition(i,:)+rand(1,nVar).*rand(1,nVar).* ...
            ((PbestPosition(jj2,:)+Gbest.Position-2*PbestPosition(i,:))+Velocity(i,:));
        DE1=PbestPosition(jj2,:)-PbestPosition(i,:);
        for ww=1:nVar
            if rand<Cr
                Position(i,ww)=PbestPosition(i,ww)+rand*rand*DE1(ww);
            end
        end
        Position(i,:)=min(max(Position(i,:),xmin),xmax);

        Cost(i)=fitness(sequence,a,lengthdata,L,Position(i,:));
        FEs=FEs+1;
        if Cost(i)>PbestCost(i)
            PbestPosition(i,:)=Position(i,:); PbestCost(i)=Cost(i); PbestVel(i,:)=Velocity(i,:);
            if PbestCost(i)>Gbest.Cost
                Gbest.Position=PbestPosition(i,:); Gbest.Cost=PbestCost(i); Gbest.Velocity=PbestVel(i,:);
            end
        end
        feBest(FEs)=Gbest.Cost;
    end
    if mod(iter,10)==0 || FEs>=MaxFEs
        fprintf('WGA: generation=%d, activePop=%d, FEs=%d/%d, best SPS=%e\n',iter,nPop,FEs,MaxFEs,Gbest.Cost);
    end
end
gbestx=Gbest.Position(:)';
gbestfitness=Gbest.Cost;
gbesthistory=msa_level_baseline_history(feBest,historyPoints);
end
