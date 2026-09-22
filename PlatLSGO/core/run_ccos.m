function [bestX,bestFitness,bestHistory] = run_ccos(suite,popsize,dimension,xmax,xmin,vmax,vmin,maxiter,f,funcId)
root = platform_root();
ccosDir = fullfile(root,'Algorithms','CCOS');
a = fullfile(ccosDir,'CEC2010','rdg','results',sprintf('F%02d.mat',funcId));
b = fullfile(ccosDir,'CEC2013','rdg','results',sprintf('F%02d.mat',funcId));
if strcmpi(suite,'CEC2010')
    src = a; dst = b;
else
    src = b; dst = a;
end
if ~exist(src,'file')
    error('CCOS decomposition file not found: %s',src);
end
backup = [tempname '.mat'];
hadDst = exist(dst,'file') == 2;
if hadDst
    copyfile(dst,backup);
end
copyfile(src,dst,'f');
cleanupObj = onCleanup(@() restore_file(dst,backup,hadDst));
addpath(ccosDir);
pathObj = onCleanup(@() rmpath(ccosDir));
[bestX,bestFitness,bestHistory] = CCOS([],popsize,dimension,xmax,xmin,vmax,vmin,maxiter,f,funcId,[]);
end

function restore_file(dst,backup,hadDst)
if hadDst
    if exist(backup,'file'), copyfile(backup,dst,'f'); delete(backup); end
else
    if exist(dst,'file'), delete(dst); end
end
end
