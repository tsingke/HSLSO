% 打分函数SPS

function sopi=SPS(testseq)
as=testseq;
[am,an]=size(as);
p=0;
for i=1:an
    for j=1:am-1
        for k=j+1:am
            if as(j,i)~=0
                if as(j,i)==as(k,i)
                    p=p+1;
                end
            end
        end
    end
end
k=0;
for i=1:an
    for j=1:am
        if as(j,i)=='_'
            k=k+1;
        end
    end
end
p=p-(am+0.5*k); 
sopi=p;
