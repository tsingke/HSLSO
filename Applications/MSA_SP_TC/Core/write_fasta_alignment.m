function write_fasta_alignment(filename, sequence, alignedSeq)
%WRITE_FASTA_ALIGNMENT Save alignment matrix as FASTA using '-' as gaps.
if iscell(alignedSeq)
    rows = alignedSeq;
else
    rows = cell(size(alignedSeq,1),1);
    for i=1:size(alignedSeq,1)
        rows{i} = alignedSeq(i,:);
    end
end
fid = fopen(filename,'w');
if fid < 0, error('Cannot create file: %s', filename); end
cleanup = onCleanup(@() fclose(fid));
lineWidth = 80;
for i=1:numel(rows)
    if isfield(sequence,'Header') && numel(sequence) >= i
        h = sequence(i).Header;
    else
        h = sprintf('seq_%d',i);
    end
    s = char(rows{i});
    s(s=='_') = '-';
    fprintf(fid,'>%s\n',h);
    for k=1:lineWidth:length(s)
        fprintf(fid,'%s\n',s(k:min(k+lineWidth-1,length(s))));
    end
end
end
