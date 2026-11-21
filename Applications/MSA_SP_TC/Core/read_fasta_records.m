function records = read_fasta_records(filename)
%READ_FASTA_RECORDS Minimal FASTA reader without Bioinformatics Toolbox.
fid = fopen(filename,'r');
if fid < 0
    error('Cannot open FASTA file: %s', filename);
end
cleanup = onCleanup(@() fclose(fid));
records = struct('Header',{},'Sequence',{});
header = '';
seq = '';
while true
    line = fgetl(fid);
    if ~ischar(line), break; end
    line = strtrim(line);
    if isempty(line), continue; end
    if line(1) == '>'
        if ~isempty(header)
            records(end+1).Header = header; %#ok<AGROW>
            records(end).Sequence = seq;
        end
        header = strtrim(line(2:end));
        seq = '';
    else
        seq = [seq regexprep(line,'\s+','')]; %#ok<AGROW>
    end
end
if ~isempty(header)
    records(end+1).Header = header;
    records(end).Sequence = seq;
end
if isempty(records)
    error('No FASTA records found in %s', filename);
end
end
