fprintf("Benchmark of single SwE computation.\n");
fprintf("Loading mock data from mock-data/*.npy...");

% Add path for loading npy files.
addpath npy-matlab/npy-matlab/

% Load mock data.
pinvDesignMtx = readNPY("../mock-data/x_pinv.npy");
residual = readNPY("../mock-data/resid.npy");
groupIds = readNPY("../mock-data/block_ids.npy");
numGroupIDs = readNPY("../mock-data/n_blocks.npy");
fprintf(" done.\n");

% Number of repetitions.
numRep = 1;

% Display parameters.
fprintf("Mock data parameters:\n");
fprintf("Number of observations: %d\n", size(residual, 1));
fprintf("Number of features: %d\n", size(residual, 2));
fprintf("Number of predictors: %d\n", size(pinvDesignMtx, 1));
fprintf("Number of blocks: %d\n", numGroupIDs);
fprintf("Number of repetitions: %d\n", numRep);

% Compute numRep versions of SwE covB serially.
fprintf("Computing SwE...");
tic;
for i=1:numRep
    covB = halfSandwichWald(pinvDesignMtx, residual, groupIds, numGroupIDs);
end
timeElapsed = toc;
fprintf(" done.\n");

% How did we do?
fprintf("Time elapsed: %f seconds.\n", timeElapsed);
fprintf("That's %f seconds per repetition.\n", timeElapsed / numRep);
