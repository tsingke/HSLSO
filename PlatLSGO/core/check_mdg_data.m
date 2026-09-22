function cleanupObj = check_mdg_data(suite,funcId)
%CHECK_MDG_DATA  Stage the DECC-MDG decomposition file for the active suite.
%
%   DECC_MDG.m reads a fixed path that always points at the CEC2013 tree
%   (Algorithms/MDG/DECC-MDG/MergedDifferentialGrouping/results2013/F%02d.mat),
%   so for a CEC2010 run the matching file has to be copied into that tree
%   first. That copy overwrites the CEC2013 file, so the original is saved and
%   restored when the returned onCleanup object goes out of scope -- i.e. when
%   the caller returns. Without the restore, running CEC2010 before CEC2013
%   (which is what main.m does) would leave CEC2010 groupings in the CEC2013
%   tree and silently corrupt every subsequent DECC_MDG CEC2013 result.
%
%   This mirrors the staging that run_ccos.m does for CCOS.
%
%   Returns [] when no swap is needed, which is the CEC2013 case: the
%   algorithm already reads that tree, so nothing is touched.

root = platform_root();
base = fullfile(root,'Algorithms','MDG','DECC-MDG','MergedDifferentialGrouping');
fname = sprintf('F%02d.mat',funcId);

if strcmpi(suite,'CEC2010')
    src = fullfile(base,'results2010',fname);
    dst = fullfile(base,'results2013',fname);
    if ~exist(src,'file')
        error('Missing DECC-MDG decomposition file: %s',src);
    end
    backup = [tempname '.mat'];
    hadDst = exist(dst,'file') == 2;
    if hadDst
        copyfile(dst,backup);
    end
    copyfile(src,dst,'f');
    cleanupObj = onCleanup(@() restore_file(dst,backup,hadDst));
else
    src = fullfile(base,'results2013',fname);
    if ~exist(src,'file')
        error('Missing DECC-MDG decomposition file: %s',src);
    end
    cleanupObj = [];
end
end

function restore_file(dst,backup,hadDst)
if hadDst
    if exist(backup,'file'), copyfile(backup,dst,'f'); delete(backup); end
else
    if exist(dst,'file'), delete(dst); end
end
end
