function [gbestx,gbestfitness,gbesthistory] = EO(sequence,a,lengthdata,L,historyPoints,dimension,MaxFEs)
% Equilibrium Optimizer MSA adaptation with exact FE accounting.
% Population size is 30. The first population is evaluated in generation 1.
if nargin<7 || isempty(MaxFEs), error('EO requires MaxFEs as input 7.'); end
if nargin<5 || isempty(historyPoints), historyPoints=200; end
lb=0; ub=1; dim=dimension; Particles_no=30; Max_iter=ceil(MaxFEs/Particles_no);
if MaxFEs<Particles_no || MaxFEs~=floor(MaxFEs), error('Invalid MaxFEs.'); end

C=zeros(Particles_no,dim);
C=initialization(Particles_no,dim,ub,lb);
C(1,:)=0.5;
Ceq1=zeros(1,dim); Ceq1_fit=-inf;
Ceq2=zeros(1,dim); Ceq2_fit=-inf;
Ceq3=zeros(1,dim); Ceq3_fit=-inf;
Ceq4=zeros(1,dim); Ceq4_fit=-inf;
a1=2; a2=1; GP=0.5; V=1;
FEs=0; feBest=nan(MaxFEs,1);
Iter=0; fit=zeros(Particles_no,1); fit_old=[]; C_old=[];

while FEs<MaxFEs && Iter<Max_iter
    Iter=Iter+1;
    for i=1:Particles_no
        if FEs>=MaxFEs, break; end
        Flag4ub=C(i,:)>ub; Flag4lb=C(i,:)<lb;
        C(i,:)=(C(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        fit(i)=fitness(sequence,a,lengthdata,L,C(i,:));
        FEs=FEs+1;

        if fit(i)>Ceq1_fit
            Ceq1_fit=fit(i); Ceq1=C(i,:);
        elseif fit(i)<Ceq1_fit && fit(i)>Ceq2_fit
            Ceq2_fit=fit(i); Ceq2=C(i,:);
        elseif fit(i)<Ceq1_fit && fit(i)<Ceq2_fit && fit(i)>Ceq3_fit
            Ceq3_fit=fit(i); Ceq3=C(i,:);
        elseif fit(i)<Ceq1_fit && fit(i)<Ceq2_fit && fit(i)<Ceq3_fit && fit(i)>Ceq4_fit
            Ceq4_fit=fit(i); Ceq4=C(i,:);
        end
        feBest(FEs)=Ceq1_fit;
    end

    if FEs>=MaxFEs
        break;
    end

    if Iter==1
        fit_old=fit; C_old=C;
    end
    for i=1:Particles_no
        if fit_old(i)>fit(i)
            fit(i)=fit_old(i); C(i,:)=C_old(i,:);
        end
    end
    C_old=C; fit_old=fit;

    Ceq_ave=(Ceq1+Ceq2+Ceq3+Ceq4)/4;
    C_pool=[Ceq1;Ceq2;Ceq3;Ceq4;Ceq_ave];
    t=(1-Iter/Max_iter)^(a2*Iter/Max_iter);

    for i=1:Particles_no
        lambda=rand(1,dim);
        lambda(lambda<eps)=eps;
        r=rand(1,dim);
        Ceq=C_pool(randi(size(C_pool,1)),:);
        F=a1*sign(r-0.5).*(exp(-lambda.*t)-1);
        r1=rand(); r2=rand();
        GCP=0.5*r1*ones(1,dim)*(r2>=GP);
        G0=GCP.*(Ceq-lambda.*C(i,:));
        G=G0.*F;
        C(i,:)=Ceq+(C(i,:)-Ceq).*F+(G./lambda*V).*(1-F);
    end

    if mod(Iter,25)==0
        fprintf('EO: generation=%d, FEs=%d/%d, best SPS=%e\n',Iter,FEs,MaxFEs,Ceq1_fit);
    end
end

gbestx=Ceq1(:)';
gbestfitness=Ceq1_fit;
gbesthistory=msa_level_baseline_history(feBest,historyPoints);
end
