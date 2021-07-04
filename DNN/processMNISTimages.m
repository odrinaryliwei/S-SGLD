function X = processMNISTimages(filename, flattened)
[fileID,errmsg] = fopen(filename,'r','b');
if fileID < 0
    error(errmsg);
end

magicNum = fread(fileID,1,'int32',0,'b');
if magicNum == 2051
    fprintf('\nRead MNIST image data...\n')
end

numImages = fread(fileID,1,'int32',0,'b');
fprintf('Number of images in the dataset: %6d ...\n',numImages);
numRows = fread(fileID,1,'int32',0,'b');
numCols = fread(fileID,1,'int32',0,'b');

X = fread(fileID,inf,'unsigned char');
X = X./255;
if flattened == 0
    X = reshape(X,numCols,numRows,1, numImages);
    X = permute(X,[2 1 3 4]);
else
    X = reshape(X,numCols*numRows,numImages);
end

fclose(fileID);
end
