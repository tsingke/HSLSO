function T = summary_stats(csvFile)
D = readtable(csvFile);
x = D.BestFitness;
x = x(isfinite(x));
T = table(mean(x),std(x),median(x),min(x),max(x),'VariableNames',{'Mean','Std','Median','Best','Worst'});
end
