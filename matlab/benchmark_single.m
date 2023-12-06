% Add path for loading npy files.
addpath npy-matlab/npy-matlab/

% Load mock data.
pinvDesignMtx = readNPY("../mock-data/x_pinv.npy");
residual = readNPY("../mock-data/resid.npy");
groupIds = readNPY("../mock-data/block_ids.npy");
numGroupIDs = readNPY("../mock-data/n_blocks.npy");

% Number of repetitions.
numRep = 1;

% Compute numRep versions of SwE covBin parallel.
tic;
for i=1:numRep
    covB = halfSandwichWald(pinvDesignMtx, residual, groupIds, numGroupIDs);
end
timeElapsed = toc;
