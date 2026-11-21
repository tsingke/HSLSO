function [bestp,bestever,gbesthistory] = SLPSO(sequence,a,lengthdata,L,historyPoints,dimension,MaxFEs)
% SL-PSO MSA adaptation with an explicit, exact fitness-evaluation budget.
% Population size is 50. The best-ranked particle is not reevaluated.
if nargin<7 || isempty(MaxFEs), error('SLPSO requires MaxFEs as input 7.'); end
m=50; d=dimension; M=100;
if MaxFEs< m || MaxFEs~=floor(MaxFEs), error('Invalid MaxFEs.'); end
if nargin<5 || isempty(historyPoints), historyPoints=200; end
c3=d/M*0.01;
PL=zeros(m,1);
for i=1:m
    PL(i)=(1-(i-1)/m)^log(sqrt(ceil(d/M)));
end
xmin=0; xmax=1;
lu=[xmin*ones(1,d);xmax*ones(1,d)];
p=lu(1,:)+rand(m,d).*(lu(2,:)-lu(1,:));
p(1,:)=0.5;
fitnessp=zeros(m,1);
v=zeros(m,d);
feBest=nan(MaxFEs,1); FEs=0;
for i=1:m
    fitnessp(i)=fitness(sequence,a,lengthdata,L,p(i,:));
    FEs=FEs+1;
    if i==1, bestever=fitnessp(i); bestp=p(i,:);
    elseif fitnessp(i)>bestever, bestever=fitnessp(i); bestp=p(i,:); end
    feBest(FEs)=bestever;
end

gen=0;
while FEs<MaxFEs
    gen=gen+1;
    [fitnessp,rank]=sort(fitnessp,'ascend');
    p=p(rank,:); v=v(rank,:);
    if fitnessp(m)>bestever
        bestever=fitnessp(m); bestp=p(m,:);
    end

    center=ones(m,1)*mean(p,1);
    randco1=rand(m,d); randco2=rand(m,d); randco3=rand(m,d);
    winidxmask=repmat((1:m)',1,d);
    winidx=winidxmask+ceil(rand(m,d).*(m-winidxmask));
    pwin=p;
    for j=1:d
        pwin(:,j)=p(winidx(:,j),j);
    end

    lpmask=repmat(rand(m,1)<PL,1,d);
    lpmask(m,:)=0;
    v1=randco1.*v+randco2.*(pwin-p)+c3*randco3.*(center-p);
    p1=p+v1;
    v=lpmask.*v1+(~lpmask).*v;
    p=lpmask.*p1+(~lpmask).*p;

    % The original SL-PSO reevaluates all m-1 non-elite particles each generation.
    for i=1:m-1
        if FEs>=MaxFEs, break; end
        p(i,:)=max(p(i,:),lu(1,:));
        p(i,:)=min(p(i,:),lu(2,:));
        fitnessp(i)=fitness(sequence,a,lengthdata,L,p(i,:));
        FEs=FEs+1;
        if fitnessp(i)>bestever
            bestp=p(i,:); bestever=fitnessp(i);
        end
        feBest(FEs)=bestever;
    end
    if mod(gen,25)==0 || FEs>=MaxFEs
        fprintf('SLPSO: generation=%d, FEs=%d/%d, best SPS=%e\n',gen,FEs,MaxFEs,bestever);
    end
end
gbesthistory=msa_level_baseline_history(feBest,historyPoints);
bestp=bestp(:)';
end
