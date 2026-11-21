function [gbestx,gbestfitness,gbesthistory] = DE(sequence,a,lengthdata,L,historyPoints,dimension,MaxFEs)
% Differential Evolution (DE/best/1/bin), MSA maximization adaptation.
if nargin<7 || isempty(MaxFEs), error('DE requires MaxFEs as input 7.'); end
if nargin<5 || isempty(historyPoints), historyPoints=200; end
popsize=50; xmin=0; xmax=1; F=0.5; CR=0.9;
if MaxFEs<popsize || MaxFEs~=floor(MaxFEs), error('Invalid MaxFEs.'); end
x=rand(popsize,dimension); x(1,:)=0.5;
v=zeros(popsize,dimension); u=zeros(popsize,dimension);
fitnessx=zeros(popsize,1); feBest=nan(MaxFEs,1); FEs=0;
gbestfitness=-inf; gbestx=[];
for i=1:popsize
    fitnessx(i)=fitness(sequence,a,lengthdata,L,x(i,:));
    FEs=FEs+1;
    if fitnessx(i)>gbestfitness
        gbestfitness=fitnessx(i); gbestx=x(i,:);
    end
    feBest(FEs)=gbestfitness;
end
iter=0;
while FEs<MaxFEs
    iter=iter+1;
    for i=1:popsize
        if FEs>=MaxFEs, break; end
        r=selectID(popsize,i,2); r1=r(1); r2=r(2);
        v(i,:)=gbestx+F*(x(r1,:)-x(r2,:));
        temp=randi(dimension,1);
        for j=1:dimension
            if rand<=CR || j==temp, u(i,j)=v(i,j);
            else, u(i,j)=x(i,j); end
        end
        u(i,:)=max(u(i,:),xmin); u(i,:)=min(u(i,:),xmax);
        ufitness=fitness(sequence,a,lengthdata,L,u(i,:));
        FEs=FEs+1;
        if ufitness>=fitnessx(i)
            fitnessx(i)=ufitness; x(i,:)=u(i,:);
        end
        if fitnessx(i)>=gbestfitness
            gbestfitness=fitnessx(i); gbestx=x(i,:);
        end
        feBest(FEs)=gbestfitness;
    end
    if mod(iter,25)==0 || FEs>=MaxFEs
        fprintf('DE: generation=%d, FEs=%d/%d, best SPS=%e\n',iter,FEs,MaxFEs,gbestfitness);
    end
end
gbesthistory=msa_level_baseline_history(feBest,historyPoints);
gbestx=gbestx(:)';
end

function r=selectID(popsize,i,count)
vec=[1:i-1,i+1:popsize];
r=zeros(1,count);
for j=1:count
    t=randi(numel(vec));
    r(j)=vec(t);
    vec(t)=[];
end
end
