% HMM模型解码问题Viterbi算法

function seq=Viterbi(sequence,a,lengthdata,A,B,L)
[m1,n1]=size(A);
[m2,n2]=size(B);
state=['M','I','D'];  %% 隐藏态
result=['A','T','C','G'];  %% 观测态
for i=1:m1
   for j=1:n1
       A(i,j)=A(i,j)/sum(A(i,:));   %% A为转移矩阵(3*m+1)行3列，每一行和为1 
   end
end
A=A'*10;  %%方便计算
    
for i=1:m2
   for j=1:n2
       B(i,j)=B(i,j)/sum(B(i,:));  %% B为状态生成矩阵(2*m+1)行4列，每一行和为1
   end
end
B=B*10;
path=[];

for l=1:lengthdata  
   m=a(1,l);
   seq=sequence(l).Sequence;  
   delta=zeros();  
   for i=1:n1-1
        if seq(1)=='A'
            t=1;
        elseif seq(1)=='G'
            t=2;
        elseif seq(1)=='C'
            t=3;
        elseif seq(1)=='T'
            t=4;
        end
        delta(i,1)=A(i)*B(i,t);
    end
    delta(n1,1)=A(n1);
    
    %递推delta矩阵剩下值
    s=ones(3,1)*(a(1,l)-1);  % 插入空格的数量
    q=ones(3,1)*(L-a(1,l));  % 剩余基因的数量
    delta_j=zeros();
    Psi=zeros();
    Psi(:,1) = 0;
    O=ones(3,1)*2;
    for k=2:L
        for j=1:3 
            if s(j,1)>0&&q(j,1)>0   
                   for i=1:n1-1
                       if seq(O(j))=='A'
                            t=1;
                        elseif seq(O(j))=='G'
                            t=2;
                        elseif seq(O(j))=='C'
                            t=3;
                        elseif seq(O(j))=='T'
                            t=4;
                       end
                       if j~=3
                            delta_j(i,1)=delta(i,k-1)*A(i,k*j)*B(k*j,t);   %若不为D，则生成观测态
                       end
                   end
              
                delta_j(n1,1)=delta(n1,k-1)*A(n1,n1*k);   %若为D，则不生成观测态
                [max_delta_j,psi]=max(delta_j);
                Psi(j,k)=psi;
                if psi~=n1
                    s(j)=s(j)-1;  
                    O(j)=O(j)+1;
                else
                    q(j)=q(j)-1;
                end
                delta(j,k)=max_delta_j;
                
                elseif s(j)>0&&q(j)==0            
                
                    for i=1:n1-1
                        if seq(O(j))=='A'
                            t=1;
                        elseif seq(O(j))=='G'
                            t=2;
                        elseif seq(O(j))=='C'
                            t=3;
                        elseif seq(O(j))=='T'
                            t=4;
                        end
                        if j~=3
                            delta_j(i,1)=delta(i,k-1)*A(i,k*j)*B(k*j,t);
                        end
                    end
                
                delta_j(n1,1)=0;
                [max_delta_j,psi]=max(delta_j);
                Psi(j,k)=psi;
                delta(j,k)=max_delta_j;
                s(j)=s(j)-1;
                O(j)=O(j)+1;
                
           elseif s(j)==0&&q(j)>0
            
                for i=1:n1-1
                    delta_j(i,1)=0;
                end
                delta_j(n1,1)=delta(n1,k-1)*A(n1,n1*k);
                [max_delta_j,psi]=max(delta_j);
                Psi(j,k)=psi;
                delta(j,k)=max_delta_j;
                q(j)=q(j)-1;
            end
                
                
            if  delta(:,k)>1000
                delta(:,k)=delta(:,k)/1000;
            end
        end
     
    end     %(for k=2:L)
    
    [P_better,psi_l] = max(delta(:,L));
    P = P_better; % 最优路径概率
    I = zeros();
    I(L,1) = psi_l;
    s1=a(1,l)-1;
    q=L-a(1,l);
    for t = L-1:-1:1
        I(t,1) = Psi(I(t+1,1),t+1); %路径回溯得到最优路径
    end
    I=I';
    path=[path;I];
end

for i=1:lengthdata
    w1=find(path(i,:)~=3);
    v1=find(path(i,:)==3);
    if length(v1)>(L-a(1,i))
        num1=length(v1)-(L-a(1,i));
        num=randperm(length(v1));
        path(i,v1(num(1:num1)))=1;
    end
    if length(v1)<(L-a(1,i))
        num1=(L-a(1,i))-length(v1);
        num=randperm(length(w1));
        path(i,w1(num(1:num1)))=3;
    end
end

for i=1:lengthdata
    w=find(path(i,:)~=3);
    v=find(path(i,:)==3); 
    seq1(i,v)='_';
    seq1(i,w)=sequence(i).Sequence;
end

u=0;
for j=1:L
    c1=find(seq1(:,j)=='_');
    if length(c1)~=lengthdata
        u=u+1;
        seq2(:,u)=seq1(:,j);
    end
end

seq=seq2;

